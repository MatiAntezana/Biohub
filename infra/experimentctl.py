#!/usr/bin/env python3
"""Compact local controller for deterministic Biohub experiment operations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aienv import ensure_aienv, in_aienv
from experimentlib import (
    ExperimentError,
    append_event,
    atomic_write_json,
    experiment_directories,
    experiment_directory,
    is_within,
    latest_events_by_run,
    load_manifest,
    load_nodes,
    load_queue,
    queue_experiment,
    read_events,
    refresh_queue,
    render_views,
    repo_root,
    run_id_for,
    set_queue_status,
    sha256_file,
    utc_now,
    validate_document_link,
    validate_kaggle_assets,
    validate_manifest,
    verify_ledger,
    write_factual_report,
)
from gpu_selector import probe_node_gpus, select_gpu
from kaggle_scores import best_by_experiment, render_leaderboard, verify_scores
from variant_planner import latest_proposals, render as render_variants, verify_ledger as verify_variant_ledger


def emit(ok: bool, action: str, **payload: Any) -> None:
    print(json.dumps({"ok": ok, "action": action, **payload}, sort_keys=True, ensure_ascii=False))


def _queue_item(experiment_id: str) -> dict[str, Any]:
    queue = refresh_queue()
    item = next((row for row in queue["items"] if row.get("experiment_id") == experiment_id), None)
    if item is None:
        raise ExperimentError(f"{experiment_id} no está en la cola; usar queue primero")
    return item


def command_doctor(_: argparse.Namespace) -> None:
    root = repo_root()
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        root / "AGENTS.md",
        root / ".agents" / "rules.md",
        root / ".agents" / "experiment-protocol.md",
        root / ".agents" / "nodes.json",
        root / ".agents" / "autonomous-exploration.json",
        root / ".agents" / "variant-recipes.json",
        root / ".agents" / "monitoring.json",
        root / "infra" / "remote_gpu_monitor.py",
        root / ".state" / "queue.json",
    ]
    errors.extend(f"falta {path.relative_to(root)}" for path in required if not path.is_file())
    errors.extend(verify_ledger(root))
    errors.extend(f"kaggle scores: {error}" for error in verify_scores(root))
    errors.extend(f"variant proposals: {error}" for error in verify_variant_ledger(root))
    if not in_aienv():
        errors.append("el controlador local no está ejecutándose dentro de AIEnv")
    nodes = load_nodes(root)
    for node_name in ("A1552", "A1556", "A1557", "A1560"):
        node = nodes.get(node_name)
        if node is None:
            errors.append(f"falta nodo remoto requerido: {node_name}")
            continue
        if not node.get("conda_env"):
            errors.append(f"{node_name} no declara conda_env")
        if node.get("status") == "OFFLINE":
            warnings.append(f"{node_name} está OFFLINE y no acepta lanzamientos")
    for experiment_id in ("EXP-030", "EXP-030B", "EXP-031"):
        errors.extend(f"{experiment_id}: {error}" for error in validate_manifest(experiment_id, root))
        errors.extend(f"{experiment_id} Kaggle: {error}" for error in validate_kaggle_assets(experiment_id, root))
    if not (root / ".git").exists():
        warnings.append("la raíz no contiene .git; gate disponible por CLI/tests, sin pre-commit")
    render_views(root)
    render_leaderboard(root)
    render_variants(root)
    emit(
        not errors,
        "doctor",
        errors=errors,
        warnings=warnings,
        experiments=len(experiment_directories(root)),
        local_conda_env=os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
    )
    if errors:
        raise SystemExit(1)


def command_validate(args: argparse.Namespace) -> None:
    errors = validate_manifest(args.experiment_id)
    emit(not errors, "validate", experiment_id=args.experiment_id, errors=errors)
    if errors:
        raise SystemExit(1)


def command_queue(args: argparse.Namespace) -> None:
    item = queue_experiment(args.experiment_id)
    emit(
        True,
        "queue",
        experiment_id=args.experiment_id,
        status=item["status"],
        auto_launch=item.get("auto_launch", False),
        node_selection=item.get("requested_node"),
    )


def command_status(_: argparse.Namespace) -> None:
    queue = refresh_queue()
    render_views()
    statuses: dict[str, int] = {}
    for item in queue["items"]:
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    emit(
        True,
        "status",
        queue=statuses,
        latest_runs=len(latest_events_by_run()),
        kaggle_top=[
            {"experiment_id": row["experiment_id"], "public_score": row["public_score"]}
            for row in best_by_experiment()[:3]
        ],
        autonomous_variants=len(latest_proposals()),
        local_conda_env=os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
    )


def command_nodes(args: argparse.Namespace) -> None:
    nodes = load_nodes()
    summary: dict[str, Any] = {}
    for name, node in sorted(nodes.items()):
        row: dict[str, Any] = {
            "status": node.get("status", "UNKNOWN"),
            "training_ready": bool(node.get("training_ready", False)),
            "conda_env": node.get("conda_env"),
            "last_verified": node.get("last_verified"),
        }
        if args.probe:
            gpus, error = probe_node_gpus(name, node)
            row["gpus"] = [gpu.as_dict() for gpu in gpus]
            row["probe_error"] = error
        summary[name] = row
    emit(True, "nodes", nodes=summary, local_conda_env=os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name)


def _node_ssh_base(node: dict[str, Any]) -> tuple[str, list[str], str]:
    host = str(node.get("host", "")).strip()
    if host.lower().startswith("a1") or ("." not in host and ":" not in host):
        target = host.lower()
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", target]
        ssh_transport = "ssh -o BatchMode=yes -o ConnectTimeout=12"
    else:
        target = f"{node['user']}@{node['host']}" if node.get("user") else node["host"]
        port = str(node.get("port", 2222))
        ssh_cmd = ["ssh", "-p", port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", target]
        ssh_transport = f"ssh -p {port} -o BatchMode=yes -o ConnectTimeout=12"
    return target, ssh_cmd, ssh_transport


def _probe_node(node_name: str, node: dict[str, Any]) -> None:
    if node.get("status") != "ONLINE" or node.get("training_ready") is not True:
        raise ExperimentError(f"{node_name} no está habilitada: status={node.get('status', 'UNKNOWN')}")
    python_check = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print('BIOHUB_REMOTE_READY')"
    )
    inner = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(node['conda_env'])} && "
        f"python -c {shlex.quote(python_check)}"
    )
    _, ssh_cmd, _ = _node_ssh_base(node)
    result = subprocess.run(
        [
            *ssh_cmd,
            "bash",
            "-lc",
            shlex.quote(inner),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or "BIOHUB_REMOTE_READY" not in result.stdout:
        raise ExperimentError(
            f"preflight remoto falló en {node_name}/{node['conda_env']}: "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )


def _snapshot(experiment_id: str, run_id: str, manifest_path: Path, manifest: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    root = repo_root()
    directory = experiment_directory(experiment_id, root)
    snapshot = directory / "code_snapshot" / run_id
    snapshot.mkdir(parents=True, exist_ok=False)
    sources = [
        manifest_path,
        directory / manifest["entrypoint"],
        root / "infra" / "experimentlib.py",
        root / "infra" / "experiment_worker.py",
        root / "src" / "strategies" / "experiment_runtime.py",
        root / "official_starter_kit" / "scripts" / "train_unet_transformer.py",
    ]
    for relative in manifest.get("code_dependencies", []):
        source = (root / relative).resolve()
        if not is_within(source, root):
            raise ExperimentError(f"code_dependency fuera del repositorio: {relative}")
        sources.append(source)
    sources = list(dict.fromkeys(sources))
    hashes: dict[str, str] = {}
    for source in sources:
        if not source.is_file():
            raise ExperimentError(f"archivo de snapshot ausente: {source}")
        relative = source.relative_to(root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256_file(source)
    atomic_write_json(snapshot / "manifest_sha256.json", hashes)
    return snapshot, hashes


def _sync_sources(node: dict[str, Any], experiment_id: str, run_id: str, log_path: Path) -> None:
    root = repo_root()
    directory = experiment_directory(experiment_id, root)
    _, manifest = load_manifest(experiment_id, root)
    run_spec = directory / "artifacts" / run_id / "run_spec.json"
    target, _, ssh_transport = _node_ssh_base(node)
    destination = f"{target}:{node['remote_root'].rstrip('/')}/"
    sources = [
        "AGENTS.md",
        ".agents/rules.md",
        ".agents/experiment-protocol.md",
        ".agents/nodes.json",
        "infra/experimentlib.py",
        "infra/experiment_worker.py",
        "src/",
        "official_starter_kit/scripts/",
        "official_starter_kit/src/",
        str(directory.relative_to(root) / "train.py"),
        str(directory.relative_to(root) / "experiment.json"),
        str(run_spec.relative_to(root)),
    ]
    variant_file = directory / "variant.json"
    if variant_file.is_file():
        sources.append(str(variant_file.relative_to(root)))
    sources.extend(
        relative for relative in manifest.get("code_dependencies", [])
        if relative not in sources and not relative.startswith("src/")
    )
    command = ["rsync", "-azR", "--exclude", "__pycache__", "-e", ssh_transport, *sources, destination]
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise ExperimentError(f"rsync falló con código {result.returncode}; ver {log_path}")


def _launch_remote(node: dict[str, Any], relative_spec: Path, log_path: Path, gpu_index: int) -> str:
    target, ssh_cmd, _ = _node_ssh_base(node)
    remote_root = node["remote_root"].rstrip("/")
    inner = (
        "set -m && "
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {shlex.quote(node['conda_env'])} && "
        f"cd ~/{shlex.quote(remote_root)} && "
        f"CUDA_VISIBLE_DEVICES={int(gpu_index)} "
        f"nohup python infra/experiment_worker.py --spec {shlex.quote(str(relative_spec))} "
        ">/dev/null 2>&1 < /dev/null & pid=$!; disown $pid; echo $pid; exit 0"
    )
    command = [
        *ssh_cmd,
        "bash",
        "-lc",
        shlex.quote(inner),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(result.stdout)
        log.write(result.stderr)
    if result.returncode:
        raise ExperimentError(f"SSH launch falló con código {result.returncode}; ver {log_path}")
    pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "unknown"
    return pid


def _start_local_watcher(run_id: str, node_name: str, log_path: Path) -> int:
    root = repo_root()
    watcher_log = log_path.parent / f"watch-{run_id}.log"
    handle = watcher_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(root / "infra" / "experiment_watch.py"), "--run-id", run_id, "--node", node_name],
        cwd=root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process.pid


def command_run(args: argparse.Namespace) -> None:
    root = repo_root()
    errors = validate_manifest(args.experiment_id, root)
    if errors:
        raise ExperimentError("; ".join(errors))
    manifest_path, manifest = load_manifest(args.experiment_id, root)
    nodes = load_nodes(root)
    execution_policy = manifest.get("execution_policy", {})
    requested_node = args.node or execution_policy.get("node_selection") or "auto"
    probes: dict[str, Any] = {}
    if requested_node == "auto" or requested_node == "max_free_vram":
        selected, probes = select_gpu(
            nodes,
            manifest["allowed_nodes"],
            min_free_vram_mib=int(execution_policy.get("min_free_vram_mib", 0)),
        )
        node_name = selected.node
        gpu_index = selected.gpu_index
        gpu_free_mib = selected.free_mib
    else:
        node_name = requested_node
        if node_name not in manifest["allowed_nodes"]:
            raise ExperimentError(f"nodo {node_name} no permitido para {args.experiment_id}")
        node_config = nodes[node_name]
        found, error = probe_node_gpus(node_name, node_config)
        probes[node_name] = {"error": error, "gpus": [gpu.as_dict() for gpu in found]}
        minimum = int(execution_policy.get("min_free_vram_mib", 0))
        eligible = [gpu for gpu in found if gpu.free_mib >= minimum]
        if not eligible:
            if args.dry_run and (node_config.get("status") != "ONLINE" or node_config.get("training_ready") is not True):
                gpu_index = None
                gpu_free_mib = None
            else:
                raise ExperimentError(f"{node_name} no tiene GPU con {minimum} MiB libres; probes={probes}")
        else:
            selected = sorted(eligible, key=lambda item: (-item.free_mib, item.utilization_pct, item.gpu_index))[0]
            gpu_index = selected.gpu_index
            gpu_free_mib = selected.free_mib
    node = nodes[node_name]
    run_id, config_hash = run_id_for(args.experiment_id, manifest)
    if args.dry_run:
        queue = load_queue(root)
        item = next((row for row in queue["items"] if row.get("experiment_id") == args.experiment_id), None)
        runnable = node.get("status") == "ONLINE" and node.get("training_ready") is True
        emit(
            True,
            "run-dry-run",
            experiment_id=args.experiment_id,
            node=node_name,
            status=item.get("status") if item else "NOT_QUEUED",
            node_status=node.get("status", "UNKNOWN"),
            remote_conda_env=node.get("conda_env"),
            runnable=runnable,
            block_reason=None if runnable else f"node status={node.get('status', 'UNKNOWN')}",
            gpu_index=gpu_index,
            gpu_free_mib=gpu_free_mib,
            selection_policy="max_free_vram" if requested_node in {"auto", "max_free_vram"} else "explicit-node",
            probes=probes,
            would_launch=False,
            config_sha256=config_hash,
        )
        return
    _probe_node(node_name, node)
    item = _queue_item(args.experiment_id)
    if item["status"] != "READY":
        raise ExperimentError(f"{args.experiment_id} está {item['status']}, no READY")

    directory = experiment_directory(args.experiment_id, root)
    run_dir = directory / "artifacts" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = directory / "logs" / f"controller-{run_id}.log"
    snapshot, hashes = _snapshot(args.experiment_id, run_id, manifest_path, manifest)
    relative_run_dir = run_dir.relative_to(root)
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": args.experiment_id,
        "node": node_name,
        "conda_env": node["conda_env"],
        "gpu_index": gpu_index,
        "gpu_free_mib_at_selection": gpu_free_mib,
        "remote_root": f"~/{node['remote_root'].lstrip('~/')}",
        "experiment_directory": str(directory.relative_to(root)),
        "entrypoint": manifest["entrypoint"],
        "arguments": manifest["arguments"],
        "run_directory": str(relative_run_dir),
        "artifact_contract": manifest["artifact_contract"],
        "data_policy": manifest["data_policy"],
        "config_sha256": config_hash,
        "snapshot_sha256": hashes,
        "prepared_at": utc_now(),
        "heartbeat_seconds": 30,
    }
    atomic_write_json(run_dir / "run_spec.json", spec)
    append_event(
        experiment_id=args.experiment_id,
        run_id=run_id,
        event_type="PREPARED",
        status="READY",
        idempotency_key=f"prepared:{run_id}",
        node=node_name,
        conda_env=node["conda_env"],
        config_sha256=config_hash,
        snapshot=str(snapshot.relative_to(root)),
    )
    try:
        _sync_sources(node, args.experiment_id, run_id, log_path)
        remote_pid = _launch_remote(node, relative_run_dir / "run_spec.json", log_path, int(gpu_index))
    except Exception as error:
        append_event(
            experiment_id=args.experiment_id,
            run_id=run_id,
            event_type="LAUNCH_FAILED",
            status="FAILED",
            idempotency_key=f"launch-failed:{run_id}",
            node=node_name,
            error=repr(error),
        )
        set_queue_status(args.experiment_id, "FAILED", run_id=run_id)
        render_views(root)
        raise
    watcher_pid = _start_local_watcher(run_id, node_name, log_path)
    set_queue_status(args.experiment_id, "RUNNING", run_id=run_id, node=node_name)
    append_event(
        experiment_id=args.experiment_id,
        run_id=run_id,
        event_type="STARTED",
        status="RUNNING",
        idempotency_key=f"started:{run_id}",
        node=node_name,
        conda_env=node["conda_env"],
        gpu_index=gpu_index,
        gpu_free_mib_at_selection=gpu_free_mib,
        remote_pid=remote_pid,
        watcher_pid=watcher_pid,
        config_sha256=config_hash,
        validation=manifest["data_policy"].get("validation", "unknown"),
    )
    render_views(root)
    emit(
        True,
        "run",
        run_id=run_id,
        experiment_id=args.experiment_id,
        status="RUNNING",
        node=node_name,
        gpu_index=gpu_index,
    )


def _sync_run_from_remote(run_id: str, node_name: str) -> Path:
    root = repo_root()
    nodes = load_nodes(root)
    node = nodes[node_name]
    event = latest_events_by_run(root).get(run_id)
    if event is None:
        raise ExperimentError(f"run_id desconocido: {run_id}")
    directory = experiment_directory(event["experiment_id"], root)
    local_run = directory / "artifacts" / run_id
    local_run.mkdir(parents=True, exist_ok=True)
    target, _, ssh_transport = _node_ssh_base(node)
    remote = (
        f"{target}:{node['remote_root'].rstrip('/')}/"
        f"{directory.relative_to(root)}/artifacts/{run_id}/"
    )
    command = [
        "rsync",
        "-az",
        "--partial",
        "-e",
        ssh_transport,
        remote,
        f"{local_run}/",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ExperimentError(f"no se pudo sincronizar {run_id}: {result.stderr.strip()}")
    return local_run


def _remote_process_alive(node_name: str, pids: list[int]) -> bool:
    if not pids:
        return False
    node = load_nodes()[node_name]
    checks = " || ".join(f"kill -0 {int(pid)} 2>/dev/null" for pid in pids)
    _, ssh_cmd, _ = _node_ssh_base(node)
    result = subprocess.run(
        [
            *ssh_cmd,
            checks,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _remote_stop_code(relative_receipt: Path) -> str:
    return (
        "import json,os,signal,pathlib,time; "
        f"p=pathlib.Path({str(relative_receipt)!r}); "
        "d=json.loads(p.read_text()) if p.is_file() else {}; "
        "pid=d.get('child_pid'); "
        "ok=False; err=None; escalated=False; exited=False; "
        "\ntry:\n"
        "    pid=int(pid); os.kill(pid, signal.SIGTERM); ok=True\n"
        "    for _ in range(40):\n"
        "        try: os.kill(pid, 0)\n"
        "        except ProcessLookupError: exited=True; break\n"
        "        time.sleep(0.5)\n"
        "    if not exited:\n"
        "        os.kill(pid, signal.SIGKILL); escalated=True; exited=True\n"
        "except Exception as exc:\n"
        "    err=repr(exc)\n"
        "print(json.dumps({'child_pid':pid,'signal':'SIGTERM','requested':ok,'exited':exited,'escalated_sigkill':escalated,'error':err}))"
    )


def _request_remote_stop(item: dict[str, Any]) -> dict[str, Any]:
    """Terminate only the recorded training child for one RUNNING queue item."""

    root = repo_root()
    experiment_id = item["experiment_id"]
    run_id = item["run_id"]
    node_name = item.get("node") or item.get("requested_node")
    nodes = load_nodes(root)
    if node_name not in nodes:
        raise ExperimentError(f"nodo desconocido para {run_id}: {node_name}")
    node = nodes[node_name]
    directory = experiment_directory(experiment_id, root)
    relative_receipt = directory.relative_to(root) / "artifacts" / run_id / "RECEIPT.json"
    python_code = _remote_stop_code(relative_receipt)
    remote_root = node["remote_root"].lstrip("~/")
    inner = f"cd ~/{shlex.quote(remote_root)} && python -c {shlex.quote(python_code)}"
    _, ssh_cmd, _ = _node_ssh_base(node)
    result = subprocess.run(
        [
            *ssh_cmd,
            "bash",
            "-lc",
            shlex.quote(inner),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ExperimentError(f"no se pudo cortar {run_id}: {(result.stderr or result.stdout).strip()[-500:]}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise ExperimentError(f"respuesta stop inválida para {run_id}: {result.stdout[-500:]}") from error
    if not payload.get("requested"):
        raise ExperimentError(f"no se señaló el child de {run_id}: {payload.get('error')}")
    return {"run_id": run_id, "experiment_id": experiment_id, "node": node_name, **payload}


def _retrieve_stopped_run(item: dict[str, Any]) -> dict[str, Any]:
    root = repo_root()
    experiment_id = item["experiment_id"]
    run_id = item["run_id"]
    node_name = item.get("node") or item.get("requested_node")
    node = load_nodes(root)[node_name]
    directory = experiment_directory(experiment_id, root)
    relative = directory.relative_to(root)
    target, _, ssh_transport = _node_ssh_base(node)
    remote_base = node["remote_root"].rstrip("/")
    transfers = [
        (f"{remote_base}/{relative}/checkpoints/{run_id}/", directory / "checkpoints" / run_id),
        (f"{remote_base}/{relative}/artifacts/{run_id}/", directory / "artifacts" / run_id),
        (f"{remote_base}/{relative}/logs/{run_id}.log", directory / "logs"),
    ]
    errors: list[str] = []
    for remote_path, local_path in transfers:
        local_path.mkdir(parents=True, exist_ok=True) if remote_path.endswith("/") else local_path.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["rsync", "-az", "--partial", "-e", ssh_transport, f"{target}:{remote_path}", f"{local_path}/"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in {0, 23}:
            errors.append((result.stderr or result.stdout).strip()[-300:])
    checkpoints = sorted((directory / "checkpoints" / run_id).rglob("*.pth"))
    return {
        "checkpoints": [str(path.relative_to(root)) for path in checkpoints],
        "transfer_errors": errors,
    }


def command_stop(args: argparse.Namespace) -> None:
    queue = load_queue()
    targets = [item for item in queue["items"] if item.get("status") == "RUNNING" and item.get("run_id")]
    if args.experiment_id:
        targets = [item for item in targets if item.get("experiment_id") == args.experiment_id]
    if not args.all and not args.experiment_id:
        raise ExperimentError("indicar EXP-XXX o --all")
    results: list[dict[str, Any]] = []
    for item in targets:
        stopped = _request_remote_stop(item)
        retrieved = _retrieve_stopped_run(item)
        append_event(
            experiment_id=item["experiment_id"],
            run_id=item["run_id"],
            event_type="STOP_REQUESTED",
            status="RUNNING",
            idempotency_key=f"stop-requested:{item['run_id']}",
            node=stopped["node"],
            child_pid=stopped.get("child_pid"),
            signal="SIGTERM",
            retrieved_checkpoints=retrieved["checkpoints"],
        )
        results.append({**stopped, **retrieved})
    emit(True, "stop", requested=len(targets), runs=results)


def reconcile_one(run_id: str, node_name: str) -> str:
    root = repo_root()
    last = latest_events_by_run(root).get(run_id)
    if last is None:
        raise ExperimentError(f"run_id desconocido: {run_id}")
    if last["status"] in {"DOC_PENDING", "CLOSED", "FAILED", "ORPHANED"}:
        return last["status"]
    run_dir = _sync_run_from_remote(run_id, node_name)
    receipt_path = run_dir / "RECEIPT.json"
    if not receipt_path.is_file():
        return "RUNNING"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    status = receipt.get("status")
    experiment_id = last["experiment_id"]
    common = {
        "node": node_name,
        "conda_env": receipt.get("environment", {}).get("conda_env"),
        "config_sha256": receipt.get("config_sha256"),
        "metrics": receipt.get("metrics", {}),
        "artifacts": receipt.get("artifacts", {}),
        "validation": receipt.get("validation", "unknown"),
        "exit_code": receipt.get("exit_code"),
    }
    if status == "SUCCEEDED":
        queue_item = next(
            (item for item in load_queue(root)["items"] if item.get("experiment_id") == experiment_id),
            {"experiment_id": experiment_id, "run_id": run_id, "node": node_name},
        )
        retrieved = _retrieve_stopped_run({**queue_item, "run_id": run_id, "node": node_name})
        common["retrieved_checkpoints"] = retrieved["checkpoints"]
        common["checkpoint_transfer_errors"] = retrieved["transfer_errors"]
        
        # Sincronización automática a Hugging Face para preservar espacio en disco
        try:
            from hf_sync import upload_experiment_checkpoints
            hf_res = upload_experiment_checkpoints(experiment_id, root=root, purge=True)
            if hf_res.get("status") == "success":
                common["hf_checkpoint_uri"] = hf_res.get("hf_uri")
        except Exception as hf_err:
            common["hf_sync_error"] = str(hf_err)

        append_event(
            experiment_id=experiment_id,
            run_id=run_id,
            event_type="COMPLETED",
            status="SUCCEEDED",
            idempotency_key=f"completed:{run_id}",
            **common,
        )
        report = write_factual_report(run_id, root)
        append_event(
            experiment_id=experiment_id,
            run_id=run_id,
            event_type="DOCUMENTATION_REQUIRED",
            status="DOC_PENDING",
            idempotency_key=f"doc-pending:{run_id}",
            report=str(report.relative_to(root)),
            **common,
        )
        set_queue_status(experiment_id, "DOC_PENDING", run_id=run_id)
        refresh_queue(root)
        render_views(root)
        return "DOC_PENDING"
    if status in {"FAILED", "ORPHANED"}:
        append_event(
            experiment_id=experiment_id,
            run_id=run_id,
            event_type=status,
            status=status,
            idempotency_key=f"terminal:{status}:{run_id}",
            error=receipt.get("error"),
            **common,
        )
        set_queue_status(experiment_id, status, run_id=run_id)
        render_views(root)
        return status
    if status == "RUNNING":
        heartbeat_path = run_dir / "HEARTBEAT.json"
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8")) if heartbeat_path.is_file() else {}
        timestamp = heartbeat.get("at")
        stale = True
        if timestamp:
            try:
                heartbeat_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                stale = (datetime.now(timezone.utc) - heartbeat_at).total_seconds() > 180
            except ValueError:
                stale = True
        if stale:
            pids = [
                int(pid)
                for pid in (receipt.get("worker_pid"), receipt.get("child_pid"), heartbeat.get("pid"))
                if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())
            ]
            if not _remote_process_alive(node_name, pids):
                append_event(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    event_type="STALE_RECONCILIATION",
                    status="ORPHANED",
                    idempotency_key=f"orphaned:{run_id}",
                    node=node_name,
                    error="heartbeat vencido y procesos remotos inexistentes",
                    heartbeat=heartbeat,
                )
                set_queue_status(experiment_id, "ORPHANED", run_id=run_id)
                render_views(root)
                return "ORPHANED"
    return "RUNNING"


def command_reconcile(args: argparse.Namespace) -> None:
    queue = load_queue()
    targets = [item for item in queue["items"] if item.get("status") == "RUNNING" and item.get("run_id")]
    if args.run_id:
        targets = [item for item in queue["items"] if item.get("run_id") == args.run_id]
    results: dict[str, str] = {}
    for item in targets:
        results[item["run_id"]] = reconcile_one(item["run_id"], item.get("node") or item["requested_node"])
    refresh_queue()
    render_views()
    emit(True, "reconcile", runs=results)


def command_close(args: argparse.Namespace) -> None:
    root = repo_root()
    latest = latest_events_by_run(root).get(args.run_id)
    if latest is None:
        raise ExperimentError(f"run_id desconocido: {args.run_id}")
    if latest["status"] != "DOC_PENDING":
        raise ExperimentError(f"{args.run_id} debe estar DOC_PENDING, está {latest['status']}")
    conclusion_path = Path(args.conclusion_file)
    if not conclusion_path.is_absolute():
        conclusion_path = root / conclusion_path
    if not conclusion_path.is_file() or not conclusion_path.read_text(encoding="utf-8").strip():
        raise ExperimentError("conclusion-file debe existir y no estar vacío")
    if not args.decision.strip():
        raise ExperimentError("decision no puede estar vacía")
    doc_link = validate_document_link(args.doc_link, root)
    append_event(
        experiment_id=latest["experiment_id"],
        run_id=args.run_id,
        event_type="CLOSED",
        status="CLOSED",
        idempotency_key=f"closed:{args.run_id}",
        conclusion_file=str(conclusion_path.resolve().relative_to(root.resolve())),
        conclusion_sha256=sha256_file(conclusion_path),
        decision=args.decision.strip(),
        documentation=doc_link,
        report=latest.get("report"),
        metrics=latest.get("metrics", {}),
        artifacts=latest.get("artifacts", {}),
        validation=latest.get("validation", "unknown"),
        node=latest.get("node"),
    )
    set_queue_status(latest["experiment_id"], "CLOSED", run_id=args.run_id)
    refresh_queue(root)
    render_views(root)
    emit(True, "close", run_id=args.run_id, status="CLOSED")


def command_import_legacy(_: argparse.Namespace) -> None:
    root = repo_root()
    imported = 0
    directories = experiment_directories(root)
    for number in range(1, 30):
        experiment_id = f"EXP-{number:03d}"
        directory = directories.get(experiment_id)
        append_event(
            experiment_id=experiment_id,
            run_id=f"legacy-{experiment_id}",
            event_type="HISTORICAL_IMPORT",
            status="IMPORTED",
            idempotency_key=f"historical-import:{experiment_id}:v1",
            provenance="repository-layout-2026-08-28",
            source=str(directory.relative_to(root)) if directory else None,
            source_missing=directory is None,
            metrics={},
            unknown_fields=["node", "conda_env", "exact_command", "artifact_hashes"],
        )
        imported += 1

    legacy_orphans = {
        "EXP-030": {
            "run_id": "legacy-EXP-030-A1556-20260827",
            "error_history": "Too many open files; a later RUNNING receipt has no live process",
            "remote_receipts": ["experiment_artifacts/EXP-030/FAILED.json", "experiment_artifacts/EXP-030/RUNNING.json"],
        },
        "EXP-031": {
            "run_id": "legacy-EXP-031-A1556-20260827",
            "error_history": "CUDA OOM; a later RUNNING receipt has no live process",
            "remote_receipts": ["experiment_artifacts/EXP-031/FAILED.json", "experiment_artifacts/EXP-031/RUNNING.json"],
        },
    }
    for experiment_id, evidence in legacy_orphans.items():
        append_event(
            experiment_id=experiment_id,
            run_id=evidence["run_id"],
            event_type="LEGACY_RECONCILIATION",
            status="ORPHANED",
            idempotency_key=f"legacy-orphan:{experiment_id}:20260827",
            node="A1556",
            provenance="read-only-ssh-inspection-2026-08-28",
            evidence=evidence,
            artifacts_preserved=True,
        )
    refresh_queue(root)
    render_views(root)
    emit(True, "import-legacy", historical=imported, orphaned=2)


def command_render(_: argparse.Namespace) -> None:
    refresh_queue()
    render_views()
    render_leaderboard()
    render_variants()
    emit(
        True,
        "render",
        files=[
            ".state/STATE.md",
            ".state/QUEUE.md",
            "experiments/RESULTS.md",
            "experiments/KAGGLE_LEADERBOARD.md",
            "experiments/AUTONOMOUS_VARIANTS.md",
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor").set_defaults(handler=command_doctor)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("experiment_id")
    validate_parser.set_defaults(handler=command_validate)
    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("experiment_id")
    queue_parser.set_defaults(handler=command_queue)
    subparsers.add_parser("status").set_defaults(handler=command_status)
    nodes_parser = subparsers.add_parser("nodes")
    nodes_parser.add_argument("--probe", action="store_true")
    nodes_parser.set_defaults(handler=command_nodes)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("experiment_id")
    run_parser.add_argument("--node")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=command_run)
    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_parser.add_argument("--run-id")
    reconcile_parser.set_defaults(handler=command_reconcile)
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("experiment_id", nargs="?")
    stop_parser.add_argument("--all", action="store_true")
    stop_parser.set_defaults(handler=command_stop)
    close_parser = subparsers.add_parser("close")
    close_parser.add_argument("run_id")
    close_parser.add_argument("--conclusion-file", required=True)
    close_parser.add_argument("--decision", required=True)
    close_parser.add_argument("--doc-link", required=True)
    close_parser.set_defaults(handler=command_close)
    subparsers.add_parser("import-legacy").set_defaults(handler=command_import_legacy)
    subparsers.add_parser("render").set_defaults(handler=command_render)
    return parser


def main() -> None:
    ensure_aienv()
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except ExperimentError as error:
        emit(False, args.command, error=str(error))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
