"""Single operator CLI: doctor, launch, status, fetch. Never publishes automatically."""
from collections import deque
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
from core.artifacts import clean, dump, code_hash, code_manifest, digest
from core.remote_executor import CONFIG, DEFAULT_NODES, execute_remotely, require_worker

CPU_TASKS = frozenset({"tests", "audit-mitosis", "evaluate-selector", "rank-repairs", "learn-events", "audit-coverage", "learn-events-015", "associate-temporal-016", "associate-temporal-017", "associate-temporal-018", "associate-temporal-019", "associate-temporal-020", "associate-temporal-021", "associate-temporal-022", "associate-temporal-023", "associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031", "associate-temporal-032", "confirm-temporal-032"})
GPU_TASKS = frozenset({"prepare", "experiment", "hoct-assets", "wheels", "wheels-py312", "package",
    "diagnose-mitosis", "learn-mitosis", "probe-mitosis", "propose-mitosis", "prepare-selector", "prepare-coverage"})
CPU_PROFILE = "cpu_evaluation_v1"
GPU_PROFILE = "gpu_exclusive_v1"
CPU_THREADS = 2
CPU_ADDRESS_SPACE_BYTES = 8 * 1024**3
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "BLIS_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2", "POLARS_MAX_THREADS": "2", "RAYON_NUM_THREADS": "2",
    "OMP_MAX_ACTIVE_LEVELS": "1", "MALLOC_ARENA_MAX": "2"}


def resource_profile(task):
    if task in CPU_TASKS:
        return CPU_PROFILE
    if task in GPU_TASKS:
        return GPU_PROFILE
    raise ValueError("Unknown task resource requirements; no implicit CPU fallback")


def child_environment(task):
    # Applied by Popen BEFORE the worker imports numpy, torch, polars or SCIP.
    env = os.environ.copy()
    env["BIOHUB_RESOURCE_PROFILE"] = resource_profile(task)
    if task in CPU_TASKS:
        env.update(CPU_ENV)
    return env


def cpu_snapshot():
    # Read only; admission headroom is NOT a reservation of host RAM/CPU.
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemAvailable", "MemTotal"}:
            amount, unit = value.split()
            if unit != "kB":
                raise ValueError("Unexpected /proc/meminfo unit")
            fields[name] = int(amount) / 1024**2
    return {"mem_available_gib": fields["MemAvailable"], "mem_total_gib": fields["MemTotal"],
        "load_1m": os.getloadavg()[0], "host_logical_cpus": os.cpu_count(),
        "allowed_cpus": sorted(os.sched_getaffinity(0))}


def resource_admission(task, environment):
    profile = resource_profile(task)
    reasons = []
    disk = environment.get("disk_free_gib")
    minimum_disk = 24 if task in {"prepare-selector", "prepare-coverage"} else 12
    if not isinstance(disk, (int, float)) or not math.isfinite(disk) or disk < minimum_disk:
        reasons.append(f"Need at least {minimum_disk} GiB free disk")
    if profile == CPU_PROFILE:
        cpu = environment.get("cpu", {})
        available, total, load, count = (cpu.get(k) for k in
            ("mem_available_gib", "mem_total_gib", "load_1m", "host_logical_cpus"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (available, total, load, count)):
            reasons.append("Missing/invalid CPU admission measurements")
        else:
            if total <= 0 or available < max(16., total * .25) or available > total:
                reasons.append("CPU evaluation needs MemAvailable >= max(16 GiB, 25% RAM)")
            if count < 1 or load < 0 or load > .75 * count:
                reasons.append("Host CPU load exceeds the conservative admission limit")
        if not cpu.get("allowed_cpus"):
            reasons.append("No allowed CPU affinity reported")
        # GPU occupancy is recorded, not waived for any CUDA-using task.
    else:
        if environment.get("gpu_processes") != "":
            reasons.append("GPU occupied; leave every existing process intact and wait")
        try:
            free = int(environment["gpu"].split(",")[-2].strip().removesuffix("MiB").strip())
        except (KeyError, IndexError, ValueError):
            free = -1
        if free < 12 * 1024:
            reasons.append("Need remote CUDA and 12 GiB free VRAM")
    return {"resource_profile": profile, "allowed": not reasons, "reasons": reasons}


def configure_cpu_worker(task):
    """Reduce only this fresh Linux worker's resources; children inherit limits."""
    import resource
    if resource_profile(task) != CPU_PROFILE or platform.system() != "Linux":
        raise RuntimeError("CPU limits apply only to approved remote CPU tasks")
    expected = {**CPU_ENV, "BIOHUB_RESOURCE_PROFILE": CPU_PROFILE}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise RuntimeError("CPU worker environment was not sealed before process startup")
    if any(name in sys.modules for name in ("torch", "numpy", "polars")):
        raise RuntimeError("CPU resource isolation must precede scientific imports")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("No CPU affinity available")
    selected = set(allowed[-CPU_THREADS:])
    os.sched_setaffinity(0, selected)
    priority = os.nice(0)
    if priority < 10:
        os.nice(10 - priority)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limits = [CPU_ADDRESS_SPACE_BYTES] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY]
    limit = min(limits)
    if limit <= 0:
        raise RuntimeError("Invalid inherited address-space limit")
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    actual = sorted(os.sched_getaffinity(0))
    if actual != sorted(selected) or os.nice(0) < 10:
        raise RuntimeError("CPU affinity/priority limits were not applied")
    return {"resource_profile": CPU_PROFILE, "logical_cpus": actual, "nice": os.nice(0),
        "address_space_limit_bytes_per_process": limit, "environment": expected}


def verify_cpu_runtime(torch):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_available()
            or torch.cuda.device_count() != 0 or torch.cuda.is_initialized()):
        raise RuntimeError("CPU worker unexpectedly exposes or initialized CUDA")
    if torch.get_num_threads() > CPU_THREADS or torch.get_num_interop_threads() > CPU_THREADS:
        raise RuntimeError("CPU worker exceeds configured torch thread limit")
    return {"cuda_visible_devices": 0, "cuda_initialized": False,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def identifier(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", value):
        raise ValueError("Use 1-40 lowercase letters/digits/hyphens for run ID")
    return value


def doctor(root, task=None):
    packages = {}
    for name in ("numpy", "scipy", "torch", "zarr", "polars", "tracksdata", "geff", "pyscipopt", "ilpy", "dask", "tqdm", "pandas", "hoct"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.executable, "code_hash": code_hash(), "packages": packages,
        "disk_free_gib": shutil.disk_usage(root).free / 1024**3,
        "cpu": cpu_snapshot(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu", "--format=csv,noheader"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "gpu_processes": subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
            check=True, capture_output=True, text=True).stdout.strip()}
    if task is not None:
        result["admission"] = resource_admission(task, result)
    return result


def launch_job(args, run, arguments):
    if args.task in {"rank-repairs", "learn-events", "prepare-coverage", "audit-coverage", "learn-events-015", "associate-temporal-016", "associate-temporal-017", "associate-temporal-018", "associate-temporal-019", "associate-temporal-020", "associate-temporal-021", "associate-temporal-022", "associate-temporal-023", "associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031"} and args.node != "A1555":
        raise ValueError("Campaign012/.../031 explicitly selects A1555; no automatic node switch")
    if run.exists():
        raise FileExistsError(f"Run already exists; inspect it instead: {run}")
    environment = doctor(args.artifact_root, args.task)
    admission = environment["admission"]
    if not admission["allowed"]:
        # No receipt, directory or process is created; this run ID remains unused.
        return {"status": "deferred_resources", "run_id": args.run_id, **admission,
            "environment": environment, "run_created": False}
    run.mkdir(parents=True, exist_ok=False)
    dump(run / "receipt.json", {"status": "queued", "task": args.task, "arm": args.arm,
        "node": args.node, "source_run": args.source_run, "tests_run": args.tests_run, "run_id": args.run_id,
        "code_hash": args.code_hash, "code_manifest": code_manifest(),
        "resource_profile": resource_profile(args.task)})
    dump(run / "admission.json", environment)
    arguments = list(arguments)
    arguments[0] = "execute"
    with (run / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *arguments],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True, env=child_environment(args.task))
    return {"status": "launched", "pid": proc.pid, "run": str(run), "code_hash": args.code_hash,
        "resource_profile": resource_profile(args.task)}


def source_run(args):
    if not args.source_run:
        raise ValueError("--source-run required")
    source = Path(args.artifact_root) / "runs" / identifier(args.source_run)
    receipt = json.loads((source / "receipt.json").read_text())
    if args.task in {"diagnose-mitosis", "learn-mitosis", "probe-mitosis", "propose-mitosis", "audit-mitosis", "prepare-selector", "evaluate-selector", "rank-repairs", "learn-events", "prepare-coverage", "audit-coverage", "learn-events-015", "associate-temporal-016", "associate-temporal-017", "associate-temporal-018", "associate-temporal-019", "associate-temporal-020", "associate-temporal-021", "associate-temporal-022", "associate-temporal-023", "associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031", "associate-temporal-032", "confirm-temporal-032"}:
        if args.task == "diagnose-mitosis":
            from core.mitosis_diagnostic import verify_diagnostic_source
            verify_diagnostic_source(receipt, code_manifest())
        elif args.task == "learn-mitosis":
            from core.mitosis_learning import verify_learning_source
            verify_learning_source(source, code_manifest())
        elif args.task == "probe-mitosis":
            from core.mitosis_probe import verify_probe_source
            verify_probe_source(source, code_manifest())
        elif args.task == "propose-mitosis":
            from core.mitosis_proposals import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "audit-mitosis":
            from core.mitosis_counterfactual import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "prepare-selector":
            from core.mitosis_selector import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "evaluate-selector":
            from core.mitosis_selector import verify_corpus
            verify_corpus(source, code_manifest())
        elif args.task == "learn-events":
            from core.event_ranking import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "prepare-coverage":
            from core.coverage_audit import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "audit-coverage":
            from core.coverage_audit import verify_corpus
            verify_corpus(source, code_manifest())
        elif args.task == "learn-events-015":
            from core.stratified_ranking_015 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-016":
            from core.temporal_association_016 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-017":
            from core.temporal_association_017 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-018":
            from core.temporal_association_018 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-019":
            from core.temporal_association_019 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-020":
            from core.temporal_association_020 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-021":
            from core.temporal_association_021 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-022":
            from core.temporal_association_022 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-023":
            from core.temporal_association_023 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-024":
            from core.temporal_association_024 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-025":
            from core.temporal_association_025 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-026":
            from core.temporal_association_026 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-027":
            from core.temporal_association_027 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-028":
            from core.temporal_association_028 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-029":
            from core.temporal_association_029 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "confirm-temporal-030":
            from core.temporal_association_030 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task == "associate-temporal-031":
            from core.temporal_association_031 import verify_sources
            verify_sources(source, code_manifest())
        elif args.task in {"associate-temporal-032", "confirm-temporal-032"}:
            from core.temporal_association_032 import verify_sources
            verify_sources(source, code_manifest())
        else:
            from core.repair_ranking import verify_sources
            verify_sources(source, code_manifest())
        if not args.tests_run:
            raise ValueError("Diagnostic requires --tests-run for the current release")
        tests = Path(args.artifact_root) / "runs" / identifier(args.tests_run)
        checked = json.loads((tests / "receipt.json").read_text())
        if (checked.get("task") != "tests" or checked.get("status") != "completed"
                or checked.get("code_hash") != args.code_hash
                or checked.get("result", {}).get("tests_passed") is not True):
            raise ValueError("Current diagnostic release tests must pass first")
        if args.task in {"rank-repairs", "learn-events", "prepare-coverage", "audit-coverage", "learn-events-015", "associate-temporal-016", "associate-temporal-017", "associate-temporal-018", "associate-temporal-019", "associate-temporal-020", "associate-temporal-021", "associate-temporal-022", "associate-temporal-023", "associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031", "associate-temporal-032", "confirm-temporal-032"}:
            r = checked["result"]
            if (checked.get("node") != args.node or args.node != "A1555"
                    or r.get("suite") != "full_discovery" or r.get("skipped") != 0
                    or r.get("errors") != 0 or r.get("failures") != 0
                    or r.get("tests_run", 0) < 168 or r.get("tests_run") != r.get("tests_expected")):
                raise ValueError("012/.../032 requires a complete zero-skip current-release suite on A1555")
            if args.task == "learn-events" and r["tests_run"] < 202:
                raise ValueError("013 cannot use the pre012 test inventory")
            if args.task in {"prepare-coverage", "audit-coverage"} and r["tests_run"] <= 247:
                raise ValueError("014 requires the expanded test inventory, not013 tests")
            if args.task == "learn-events-015" and r["tests_run"] < 277:
                raise ValueError("015 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-016" and r["tests_run"] < 295:
                raise ValueError("016 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-017" and r["tests_run"] < 299:
                raise ValueError("017 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-018" and r["tests_run"] < 304:
                raise ValueError("018 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-019" and r["tests_run"] < 310:
                raise ValueError("019 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-020" and r["tests_run"] < 320:
                raise ValueError("020 requires the full expanded test inventory, not historical tests")
            if args.task == "associate-temporal-021" and r["tests_run"] < 330:
                raise ValueError("021 requires the full expanded test inventory, not historical tests")
            if args.task in {"associate-temporal-022", "associate-temporal-023"} and r["tests_run"] < 340:
                raise ValueError("022/023 requires the full expanded test inventory, not historical tests")
            if args.task in {"associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031", "associate-temporal-032", "confirm-temporal-032"} and r["tests_run"] < 350:
                raise ValueError("024/.../032 requires the full expanded test inventory, not historical tests")
        return source

    if args.task == "package":
        if receipt["status"] != "completed":
            raise ValueError("Source run must be complete")
        if receipt["task"] in {"confirm-temporal-030", "confirm-temporal-032"}:
            if receipt.get("result", {}).get("disposition") != "eligible_for_final_package_and_submission":
                raise ValueError("Temporal confirmation run did not pass confirmation gate")
            if not args.tests_run:
                raise ValueError("Package requires --tests-run for the current release")
            tests = Path(args.artifact_root) / "runs" / identifier(args.tests_run)
            checked = json.loads((tests / "receipt.json").read_text())
            if (checked.get("task") != "tests" or checked.get("status") != "completed"
                    or checked.get("code_hash") != args.code_hash
                    or checked.get("result", {}).get("tests_passed") is not True):
                raise ValueError("Current release tests must pass first")
            return source
        elif receipt["task"] == "experiment":
            if receipt["code_hash"] != args.code_hash:
                raise ValueError("Source run must be from the identical frozen code release")
            return source
        else:
            raise ValueError(f"Expected source task experiment or confirm-temporal-*, got {receipt['task']}")

    if receipt["status"] != "completed" or receipt["code_hash"] != args.code_hash:
        raise ValueError("Source run must be complete and from the identical frozen code release")
    required = {"prepare": "tests", "experiment": "prepare"}[args.task]
    if receipt["task"] != required:
        raise ValueError(f"Expected source task {required}")
    return source


def execute_job(args, run):
    import fcntl
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(pid=os.getpid(), status="running", started_unix=time.time())
    dump(receipt_path, receipt)
    try:
        cpu_only = resource_profile(args.task) == CPU_PROFILE
        if cpu_only:
            isolation = configure_cpu_worker(args.task)
            dump(run / "cpu-isolation.json", isolation)
        hoct_site = Path(args.artifact_root) / "assets/hoct-site"
        if hoct_site.is_dir():
            sys.path.insert(0, str(hoct_site))
        with (Path(args.artifact_root) / "gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            environment = doctor(args.artifact_root)
            dump(run / "environment.json", environment)
            admission = resource_admission(args.task, environment)
            if not admission["allowed"]:
                raise RuntimeError("; ".join(admission["reasons"]))
            import torch
            if cpu_only:
                torch.set_num_threads(CPU_THREADS)
                torch.set_num_interop_threads(1)
                dump(run / "cpu-runtime-before.json", verify_cpu_runtime(torch))
            elif not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
                raise RuntimeError("Need remote CUDA and 12 GiB free VRAM")
            from ops.workflow import prepare_job, experiment_job
            if args.task == "tests":
                test_report = run / "tests.json"
                completed = subprocess.run([sys.executable, str(ROOT / "ops/verify.py"),
                    "--full", "--report", str(test_report)], cwd=ROOT)
                if completed.returncode:
                    raise RuntimeError("Tests failed")
                result = json.loads(test_report.read_text())
                if (result.get("tests_passed") is not True or result.get("skipped") != 0
                        or result.get("tests_run", 0) < 119 or result.get("suite") != "full_discovery"):
                    raise RuntimeError("Incomplete or skipped remote full test suite")
            elif args.task == "hoct-assets":
                from ops.external_assets import hoct_job
                result = hoct_job(args, run)
            elif args.task in {"wheels", "wheels-py312"}:
                from ops.bundle_build import wheelhouse_job
                result = wheelhouse_job(args, run)
            elif args.task == "prepare":
                source_run(args)
                result = prepare_job(args, run)
            elif args.task == "experiment":
                result = experiment_job(args, run, source_run(args))
            elif args.task == "diagnose-mitosis":
                from core.mitosis_diagnostic import diagnostic_job
                result = diagnostic_job(args, run, source_run(args))
            elif args.task == "learn-mitosis":
                from core.mitosis_learning import learning_job
                result = learning_job(args, run, source_run(args))
            elif args.task == "probe-mitosis":
                from core.mitosis_probe import probe_job
                result = probe_job(args, run, source_run(args))
            elif args.task == "propose-mitosis":
                from core.mitosis_proposals import proposal_job
                result = proposal_job(args, run, source_run(args))
            elif args.task == "audit-mitosis":
                from core.mitosis_counterfactual import counterfactual_job
                result = counterfactual_job(args, run, source_run(args))
            elif args.task == "prepare-selector":
                from core.mitosis_selector import prepare_job as prepare_selector
                result = prepare_selector(args, run, source_run(args))
            elif args.task == "evaluate-selector":
                from core.mitosis_selector import evaluate_job as evaluate_selector
                result = evaluate_selector(args, run, source_run(args))
            elif args.task == "rank-repairs":
                from core.repair_ranking import evaluate_job as rank_repairs
                result = rank_repairs(args, run, source_run(args))
            elif args.task == "learn-events":
                from core.event_ranking import assay_job
                result = assay_job(args, run, source_run(args))
            elif args.task == "prepare-coverage":
                from core.coverage_audit import prepare_job as prepare_coverage
                result = prepare_coverage(args, run, source_run(args))
            elif args.task == "audit-coverage":
                from core.coverage_audit import audit_job as audit_coverage
                result = audit_coverage(args, run, source_run(args))
            elif args.task == "learn-events-015":
                from core.stratified_ranking_015 import assay_job
                result = assay_job(args, run, source_run(args))
            elif args.task == "associate-temporal-016":
                from core.temporal_association_016 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-017":
                from core.temporal_association_017 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-018":
                from core.temporal_association_018 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-019":
                from core.temporal_association_019 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-020":
                from core.temporal_association_020 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-021":
                from core.temporal_association_021 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-022":
                from core.temporal_association_022 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-023":
                from core.temporal_association_023 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-024":
                from core.temporal_association_024 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-025":
                from core.temporal_association_025 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-026":
                from core.temporal_association_026 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-027":
                from core.temporal_association_027 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-028":
                from core.temporal_association_028 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-029":
                from core.temporal_association_029 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "confirm-temporal-030":
                from core.temporal_association_030 import temporal_confirmation_job
                result = temporal_confirmation_job(args, run, source_run(args))
            elif args.task == "associate-temporal-031":
                from core.temporal_association_031 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "associate-temporal-032":
                from core.temporal_association_032 import temporal_association_job
                result = temporal_association_job(args, run, source_run(args))
            elif args.task == "confirm-temporal-032":
                from core.temporal_association_032 import temporal_confirmation_job
                result = temporal_confirmation_job(args, run, source_run(args))
            else:
                from ops.bundle_build import package_job
                result = package_job(args, run, source_run(args))
            if cpu_only:
                dump(run / "cpu-runtime-after.json", verify_cpu_runtime(torch))
            result["resource_profile"] = resource_profile(args.task)
        receipt.update(status="completed", result=result)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["finished_unix"] = time.time()
        dump(receipt_path, receipt)


def fetch_destination(destination, node, run_id, resume):
    """Validate an explicit retrieval target; resume never authorizes overwrites."""
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("Refuse symlink retrieval destination")
    if not resume:
        destination.mkdir(parents=True, exist_ok=False)
        return {}
    if not destination.is_dir() or any(p.is_symlink() for p in destination.rglob("*")):
        raise ValueError("Resume requires an existing plain run directory")
    receipt = json.loads((destination/"receipt.json").read_text())
    if (receipt.get("node") != node or receipt.get("run_id") != run_id
            or receipt.get("status") not in {"completed", "failed"}):
        raise ValueError("Resume requires the matching terminal local receipt")
    return {p: digest(p) for p in destination.rglob("*") if p.is_file()}


def fetch_command(node, run_id, destination, resume):
    command = ["rsync", "-az", "--no-links", "--partial-dir=.biohub-rsync-partial"]
    if resume:
        command.append("--ignore-existing")
    return command + ["-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
        f"{node.ssh_alias}:{CONFIG['remote_root']}/runs/{run_id}/", str(destination)+"/"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["doctor", "launch", "execute", "status", "fetch"])
    p.add_argument("--node", choices=sorted(DEFAULT_NODES), default=CONFIG["default_node"])
    p.add_argument("--run-id")
    p.add_argument("--task", choices=sorted(CPU_TASKS | GPU_TASKS))
    p.add_argument("--resume", action="store_true", help="fetch only: add missing files; preserve every existing local file")
    p.add_argument("--source-run")
    p.add_argument("--tests-run", help="Current-release tests receipt for diagnostic-only reuse")
    p.add_argument("--arm", choices=["mitosis_add", "hoct", "hoct_mitosis", "hoct_rescue", "full"], default="mitosis_add")
    p.add_argument("--data-root", default=CONFIG["data_root"])
    p.add_argument("--legacy-root", default=CONFIG["legacy_assets"])
    p.add_argument("--worker", action="store_true")
    p.add_argument("--artifact-root", default=CONFIG["remote_root"])
    p.add_argument("--code-hash")
    args = p.parse_args()
    if args.resume and (args.action != "fetch" or args.worker):
        p.error("--resume is only available for local retrieval")
    if args.task in {"rank-repairs", "learn-events", "prepare-coverage", "audit-coverage", "learn-events-015", "associate-temporal-016", "associate-temporal-017", "associate-temporal-018", "associate-temporal-019", "associate-temporal-020", "associate-temporal-021", "associate-temporal-022", "associate-temporal-023", "associate-temporal-024", "associate-temporal-025", "associate-temporal-026", "associate-temporal-027", "associate-temporal-028", "associate-temporal-029", "confirm-temporal-030", "associate-temporal-031", "associate-temporal-032", "confirm-temporal-032"} and args.node != "A1555":
        p.error("Campaign012/.../032 requires explicit --node A1555")
    if args.action != "doctor":
        if not args.run_id:
            p.error("--run-id required")
        identifier(args.run_id)
    if args.action in {"launch", "execute"} and not args.task:
        p.error("--task required")
    if not args.worker:
        if args.action == "execute":
            p.error("execute is worker-internal")
        if args.action == "fetch":
            destination = ROOT / "artifacts" / args.node / args.run_id
            if not args.resume and destination.exists():
                raise FileExistsError("Local run already exists; use --resume only for a terminal partial retrieval")
            existing = fetch_destination(destination, args.node, args.run_id, True) if args.resume else {}
            execute_remotely(__file__, DEFAULT_NODES[args.node], ["status", "--run-id", args.run_id])
            if not args.resume:
                fetch_destination(destination, args.node, args.run_id, False)
            # Retrieval only: exact run subtree, never writes back to a worker.
            subprocess.run(fetch_command(DEFAULT_NODES[args.node], args.run_id, destination, args.resume), check=True)
            if any(not path.is_file() or digest(path) != sha for path, sha in existing.items()):
                raise ValueError("Existing local evidence changed during retrieval; stop and report")
            print(destination)
            return
        arguments = [args.action]
        for key in ("run_id", "task", "source_run", "tests_run", "arm", "data_root", "legacy_root"):
            value = getattr(args, key)
            if value is not None:
                arguments += ["--" + key.replace("_", "-"), value]
        execute_remotely(__file__, DEFAULT_NODES[args.node], arguments)
        return
    require_worker(DEFAULT_NODES[args.node])
    if args.code_hash != code_hash():
        raise ValueError("Release code differs from the dispatched hash")
    root = Path(args.artifact_root)
    if root != Path(CONFIG["remote_root"]):
        raise ValueError("Artifact root differs from isolated project configuration")
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "doctor":
        print(json.dumps(clean(doctor(root, args.task)), indent=2))
        return
    run = root / "runs" / args.run_id
    if args.action == "launch":
        print(json.dumps(clean(launch_job(args, run, sys.argv[1:])), indent=2))
    elif args.action == "execute":
        execute_job(args, run)
    else:
        receipt = json.loads((run / "receipt.json").read_text())
        with (run / "worker.log").open(errors="replace") as log:
            tail = list(deque(log, maxlen=12))
        print(json.dumps(clean({k: v for k, v in receipt.items() if k != "code_manifest"}), indent=2))
        print("".join(tail)[-12000:])


if __name__ == "__main__":
    main()
