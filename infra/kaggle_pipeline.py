#!/usr/bin/env python3
"""Policy-gated Kaggle push, log monitoring, output validation and submission.

Normal output is one compact JSON line. Full Kaggle stdout/stderr and notebook
logs are persisted below ``.state/kaggle/`` so they do not pollute agent context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from aienv import ensure_aienv
from experimentlib import (
    SUBMISSION_COLUMNS,
    ExperimentError,
    atomic_write_json,
    atomic_write_text,
    exclusive_lock,
    experiment_directory,
    load_manifest,
    read_json,
    repo_root,
    utc_now,
    validate_kaggle_assets,
)
from kaggle_scores import record_job_score


TERMINAL_JOB_STATES = {"SCORED", "COMPLETE_VALIDATED", "POLICY_BLOCKED", "NEEDS_REPAIR", "FAILED"}
TRANSIENT_ERRORS = (
    "429",
    "too many requests",
    "service unavailable",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "worker lost",
    "internal server error",
    "gateway timeout",
)
DETERMINISTIC_ERRORS = ("traceback", "cuda out of memory", "memoryerror", "modulenotfounderror", "filenotfounderror")
TRAINING_PATTERNS = (
    re.compile(r"\boptimizer\.(?:step|zero_grad)\s*\("),
    re.compile(r"\.backward\s*\("),
    re.compile(r"\bmodel\.train\s*\("),
    re.compile(r"\b(?:fit|fit_generator)\s*\("),
)
TRAIN_PATH_PATTERN = re.compile(r"(?:^|[/\\])train(?:$|[/\\])", re.IGNORECASE)
SCOPE_MARKER = "BIOHUB_DATASET_SCOPE="
PATH_MARKER = "BIOHUB_DATASET_PATH="
TEST_GUARD_MARKER = "BIOHUB_TEST_ONLY_GUARD"


def _compact(ok: bool, action: str, **payload: Any) -> None:
    print(json.dumps({"ok": ok, "action": action, **payload}, sort_keys=True, ensure_ascii=False))


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    result = runner(command, cwd=cwd, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    block = (
        f"\n[{utc_now()}] $ {' '.join(command)}\n"
        f"[stdout]\n{result.stdout or ''}\n[stderr]\n{result.stderr or ''}\n"
    )
    atomic_write_text(log_path, existing + block)
    return result


def package_hash(kaggle_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(kaggle_dir.glob("*")):
        if path.is_file() and path.name not in {".DS_Store"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def kaggle_config(experiment_id: str, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = root or repo_root()
    directory = experiment_directory(experiment_id, root)
    try:
        _, manifest = load_manifest(experiment_id, root)
        config = manifest.get("kaggle", {})
    except ExperimentError:
        config = read_json(directory / "kaggle" / "deployment.json", {})
    if config.get("enabled") is not True:
        raise ExperimentError(f"Kaggle no está habilitado para {experiment_id}")
    if config.get("deployment_ready") is not True:
        raise ExperimentError(f"paquete Kaggle de {experiment_id} todavía no está marcado deployment_ready")
    return directory / "kaggle", config


def static_preflight(experiment_id: str, root: Path | None = None) -> list[str]:
    root = root or repo_root()
    errors = list(validate_kaggle_assets(experiment_id, root))
    try:
        kaggle_dir, config = kaggle_config(experiment_id, root)
    except ExperimentError as error:
        return errors + [str(error)]
    inference = kaggle_dir / "inference.py"
    metadata = read_json(kaggle_dir / "kernel-metadata.json", {})
    source = inference.read_text(encoding="utf-8") if inference.is_file() else ""
    if config.get("training_allowed") is not False:
        errors.append("kaggle.training_allowed debe ser false")
    if config.get("test_only") is not True:
        errors.append("kaggle.test_only debe ser true")
    if not config.get("competition"):
        errors.append("kaggle.competition es obligatorio")
    if not config.get("kernel_slug"):
        errors.append("kaggle.kernel_slug es obligatorio")
    if metadata.get("id") != config.get("kernel_slug"):
        errors.append("kernel_slug no coincide con kernel-metadata.json")
    if config.get("output_file") != "submission.csv":
        errors.append("kaggle.output_file debe ser submission.csv")
    if TEST_GUARD_MARKER not in source or SCOPE_MARKER not in source or PATH_MARKER not in source:
        errors.append("inference.py no contiene la guardia estructurada test-only")
    if any(pattern.search(source) for pattern in TRAINING_PATTERNS):
        errors.append("inference.py contiene una operación de entrenamiento prohibida")
    # Detect literal train dataset paths without flagging prose such as 'training'.
    for literal in re.findall(r"['\"]([^'\"]+)['\"]", source):
        if TRAIN_PATH_PATTERN.search(literal.replace("-training", "")):
            errors.append(f"inference.py referencia ruta train prohibida: {literal}")
            break
    return sorted(set(errors))


def inspect_runtime_log(text: str, *, require_scope: bool = False) -> dict[str, Any]:
    scopes: list[dict[str, Any]] = []
    paths: list[str] = []
    violations: list[str] = []

    extracted_lines: list[str] = []
    for line in text.splitlines():
        trimmed = line.strip().lstrip(",").strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                obj = json.loads(trimmed)
                if isinstance(obj, dict) and "data" in obj:
                    extracted_lines.extend(str(obj["data"]).splitlines())
                    continue
            except Exception:
                pass
        extracted_lines.append(line)

    for line in extracted_lines:
        if SCOPE_MARKER in line:
            raw = line.split(SCOPE_MARKER, 1)[1].strip()
            try:
                scopes.append(json.loads(raw))
            except json.JSONDecodeError:
                start_b = raw.find("{")
                end_b = raw.rfind("}")
                if start_b != -1 and end_b != -1 and end_b > start_b:
                    try:
                        scopes.append(json.loads(raw[start_b : end_b + 1]))
                    except json.JSONDecodeError:
                        violations.append("scope sentinel JSON inválido")
                else:
                    violations.append("scope sentinel JSON inválido")
        if PATH_MARKER in line:
            path = line.split(PATH_MARKER, 1)[1].strip()
            paths.append(path)
            if TRAIN_PATH_PATTERN.search(path):
                violations.append(f"se intentó procesar train: {path}")
    for scope in scopes:
        if scope.get("scope") != "test":
            violations.append(f"scope no permitido: {scope.get('scope')}")
        for path in scope.get("paths", []):
            if TRAIN_PATH_PATTERN.search(str(path)):
                violations.append(f"scope contiene train: {path}")
    if require_scope and not scopes:
        violations.append("la corrida terminó sin BIOHUB_DATASET_SCOPE=test")
    lower = text.lower()
    has_error = any(token in lower for token in DETERMINISTIC_ERRORS)
    transient = any(token in lower for token in TRANSIENT_ERRORS)
    return {
        "scopes": scopes,
        "paths": paths,
        "violations": sorted(set(violations)),
        "has_error": has_error,
        "transient_error": transient,
    }


def validate_submission_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(f"output ausente: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise ExperimentError(f"columnas CSV inválidas: {reader.fieldnames}")
        row_count = 0
        node_count = 0
        edge_count = 0
        ids: set[str] = set()
        for row in reader:
            row_count += 1
            if row["id"] in ids:
                raise ExperimentError(f"id CSV duplicado: {row['id']}")
            ids.add(row["id"])
            if row["row_type"] == "node":
                node_count += 1
            elif row["row_type"] == "edge":
                edge_count += 1
            else:
                raise ExperimentError(f"row_type inválido: {row['row_type']}")
            if TRAIN_PATH_PATTERN.search(row["dataset"]):
                raise ExperimentError(f"dataset CSV parece train: {row['dataset']}")
    if row_count == 0 or node_count == 0:
        raise ExperimentError("submission.csv está vacío o no contiene nodos")
    return {"rows": row_count, "nodes": node_count, "edges": edge_count, "bytes": path.stat().st_size}


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or repo_root()
        self.path = self.root / ".state" / "kaggle-jobs.json"
        self.lock = self.root / ".state" / "kaggle-jobs.lock"

    def read(self) -> dict[str, Any]:
        return read_json(self.path, {"schema_version": 1, "jobs": {}})

    def save_job(self, job: dict[str, Any]) -> None:
        with exclusive_lock(self.lock):
            payload = self.read()
            payload.setdefault("jobs", {})[job["job_id"]] = job
            payload["updated_at"] = utc_now()
            atomic_write_json(self.path, payload)

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.read().get("jobs", {}).get(job_id)
        if not job:
            raise ExperimentError(f"job Kaggle desconocido: {job_id}")
        return job


class KagglePipeline:
    def __init__(
        self,
        root: Path | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.root = root or repo_root()
        self.runner = runner
        self.store = JobStore(self.root)

    def _command(self, job: dict[str, Any], command: list[str], label: str) -> subprocess.CompletedProcess[str]:
        job_dir = self.root / ".state" / "kaggle" / job["job_id"]
        return _run(command, cwd=self.root, log_path=job_dir / f"{label}.log", runner=self.runner)

    def preflight(self, experiment_id: str, *, build: bool = True) -> dict[str, Any]:
        kaggle_dir, config = kaggle_config(experiment_id, self.root)
        if build:
            builder = kaggle_dir / "build_nb.py"
            if not builder.is_file():
                raise ExperimentError("falta kaggle/build_nb.py")
            temp_job = {"job_id": f"preflight-{experiment_id}"}
            result = self._command(temp_job, [sys.executable, str(builder)], "build")
            if result.returncode:
                raise ExperimentError("build_nb.py falló; ver .state/kaggle/preflight-*/build.log")
        errors = static_preflight(experiment_id, self.root)
        if errors:
            raise ExperimentError("; ".join(errors))
        return {
            "experiment_id": experiment_id,
            "kernel": config["kernel_slug"],
            "competition": config["competition"],
            "package_sha256": package_hash(kaggle_dir),
        }

    def push(self, experiment_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        check = self.preflight(experiment_id)
        kaggle_dir, config = kaggle_config(experiment_id, self.root)
        job_id = f"{experiment_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
        job: dict[str, Any] = {
            "schema_version": 1,
            "job_id": job_id,
            "experiment_id": experiment_id,
            "kernel_slug": config["kernel_slug"],
            "competition": config["competition"],
            "output_file": config.get("output_file", "submission.csv"),
            "auto_submit": config.get("auto_submit") is True,
            "max_retries": int(config.get("max_retries", 1)),
            "retries": 0,
            "state": "DRY_RUN" if dry_run else "PUSHING",
            "created_at": utc_now(),
            "package_sha256": check["package_sha256"],
        }
        self.store.save_job(job)
        if dry_run:
            return job
        result = self._command(
            job,
            ["kaggle", "kernels", "push", "-p", str(kaggle_dir), "--accelerator", "NvidiaTeslaT4"],
            "push",
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"Kernel version\s+(\d+)\s+successfully pushed", output, re.IGNORECASE)
        if result.returncode or not match:
            job.update(state="FAILED", error=output.strip()[-1000:], updated_at=utc_now())
        else:
            job.update(state="RUNNING", kernel_version=int(match.group(1)), pushed_at=utc_now(), updated_at=utc_now())
        self.store.save_job(job)
        return job

    def _status_and_logs(self, job: dict[str, Any]) -> tuple[str, str]:
        status_result = self._command(
            job, ["kaggle", "kernels", "status", job["kernel_slug"]], "status-history"
        )
        status_text = (status_result.stdout or status_result.stderr).strip()
        upper = status_text.upper()
        state = next(
            (candidate for candidate in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "RUNNING", "QUEUED") if candidate in upper),
            "UNKNOWN",
        )
        logs_result = self._command(job, ["kaggle", "kernels", "logs", job["kernel_slug"]], "logs-history")
        logs = logs_result.stdout or logs_result.stderr or ""
        job_dir = self.root / ".state" / "kaggle" / job["job_id"]
        atomic_write_text(job_dir / "notebook.log", logs)
        return state, logs

    def _retry(self, job: dict[str, Any]) -> dict[str, Any]:
        kaggle_dir, _ = kaggle_config(job["experiment_id"], self.root)
        result = self._command(
            job,
            ["kaggle", "kernels", "push", "-p", str(kaggle_dir), "--accelerator", "NvidiaTeslaT4"],
            f"retry-{job['retries'] + 1}",
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"Kernel version\s+(\d+)\s+successfully pushed", output, re.IGNORECASE)
        job["retries"] += 1
        if result.returncode or not match:
            job.update(state="FAILED", error=output.strip()[-1000:])
        else:
            job.update(state="RUNNING", kernel_version=int(match.group(1)), last_retry_at=utc_now())
        job["updated_at"] = utc_now()
        self.store.save_job(job)
        return job

    def _download_validate_submit(self, job: dict[str, Any]) -> dict[str, Any]:
        job_dir = self.root / ".state" / "kaggle" / job["job_id"]
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self._command(
            job,
            ["kaggle", "kernels", "output", job["kernel_slug"], "-p", str(output_dir), "-o", "-q"],
            "output",
        )
        if result.returncode:
            job.update(state="FAILED", error=(result.stderr or result.stdout).strip()[-1000:])
            return job
        output_path = output_dir / job["output_file"]
        try:
            job["csv_validation"] = validate_submission_csv(output_path)
        except ExperimentError as error:
            job.update(state="POLICY_BLOCKED", error=str(error))
            return job
        if not job.get("auto_submit"):
            job["state"] = "COMPLETE_VALIDATED"
            return job
        message = f"{job['experiment_id']} auto {job['job_id']}"
        submit = self._command(
            job,
            [
                "kaggle",
                "competitions",
                "submit",
                job["competition"],
                "-k",
                job["kernel_slug"],
                "-v",
                str(job["kernel_version"]),
                "-f",
                job["output_file"],
                "-m",
                message,
            ],
            "submit",
        )
        if submit.returncode:
            job.update(state="FAILED", error=(submit.stderr or submit.stdout).strip()[-1000:])
        else:
            job.update(state="SUBMITTED", submission_message=message, submission_response=submit.stdout.strip()[-1000:])
        return job

    def _poll_score(self, job: dict[str, Any]) -> dict[str, Any]:
        result = self._command(
            job,
            ["kaggle", "competitions", "submissions", job["competition"], "--csv", "--page-size", "50"],
            "scores-history",
        )
        if result.returncode:
            return job
        clean_lines = [line for line in result.stdout.splitlines() if not line.lower().startswith("warning:")]
        rows = list(csv.DictReader(io.StringIO("\n".join(clean_lines))))
        match = next(
            (row for row in rows if job.get("submission_message") in (row.get("description") or row.get("message") or "")),
            rows[0] if rows else None,
        )
        if match:
            clean_match = {str(k): v for k, v in match.items() if k is not None}
            job["submission"] = clean_match
            score = clean_match.get("publicScore") or clean_match.get("public_score")
            if score not in (None, "", "None"):
                job["state"] = "SCORED"
                job["public_score"] = float(score)
                record_job_score(job, clean_match, self.root)
        return job

    def watch_once(self, job_id: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if job["state"] == "SUBMITTED":
            job = self._poll_score(job)
        elif job["state"] in {"RUNNING", "QUEUED", "UNKNOWN"}:
            kernel_state, logs = self._status_and_logs(job)
            inspection = inspect_runtime_log(logs, require_scope=kernel_state == "COMPLETE")
            job.update(kernel_state=kernel_state, log_inspection=inspection)
            if inspection["violations"]:
                # Kaggle's public CLI has no safe session-cancel endpoint. The
                # in-notebook guard raises before inference; externally we stop
                # retries and submissions immediately.
                job.update(state="POLICY_BLOCKED", error="; ".join(inspection["violations"]))
            elif kernel_state == "COMPLETE":
                job = self._download_validate_submit(job)
            elif kernel_state in {"ERROR", "CANCEL_ACKNOWLEDGED"}:
                if inspection["transient_error"] and job["retries"] < job["max_retries"]:
                    job = self._retry(job)
                else:
                    job.update(state="NEEDS_REPAIR", error="Kaggle kernel failed; inspect persisted notebook.log")
            else:
                job["state"] = kernel_state
        job["updated_at"] = utc_now()
        self.store.save_job(job)
        return job


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("experiment_id")
    push = sub.add_parser("push")
    push.add_argument("experiment_id")
    push.add_argument("--dry-run", action="store_true")
    watch = sub.add_parser("watch")
    watch.add_argument("job_id")
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--interval", type=int, default=60)
    status = sub.add_parser("status")
    status.add_argument("--job-id")
    args = parser.parse_args()
    pipeline = KagglePipeline()
    try:
        if args.command == "preflight":
            result = pipeline.preflight(args.experiment_id)
            _compact(True, "kaggle-preflight", **result)
            return
        if args.command == "push":
            job = pipeline.push(args.experiment_id, dry_run=args.dry_run)
            _compact(job["state"] not in {"FAILED", "POLICY_BLOCKED"}, "kaggle-push", **job)
            return
        if args.command == "status":
            payload = pipeline.store.get(args.job_id) if args.job_id else pipeline.store.read()
            _compact(True, "kaggle-status", payload=payload)
            return
        while True:
            job = pipeline.watch_once(args.job_id)
            _compact(job["state"] not in {"FAILED", "POLICY_BLOCKED", "NEEDS_REPAIR"}, "kaggle-watch", **job)
            if args.once or job["state"] in TERMINAL_JOB_STATES:
                return
            time.sleep(max(15, args.interval))
    except ExperimentError as error:
        _compact(False, f"kaggle-{args.command}", error=str(error))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
