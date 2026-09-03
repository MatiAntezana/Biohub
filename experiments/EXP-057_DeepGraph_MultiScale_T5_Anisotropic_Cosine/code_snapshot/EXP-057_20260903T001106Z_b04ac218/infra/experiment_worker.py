#!/usr/bin/env python3
"""Remote worker: execute one frozen Biohub run and emit atomic receipts."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experimentlib import ExperimentError, atomic_write_json, exclusive_lock, sha256_file, utc_now


def _environment(spec: dict[str, Any]) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu.returncode:
        raise ExperimentError("nvidia-smi no está disponible; se rechaza cómputo no-GPU")
    active_conda = os.environ.get("CONDA_DEFAULT_ENV")
    expected_conda = spec.get("conda_env")
    if expected_conda and active_conda != expected_conda:
        raise ExperimentError(f"entorno Conda activo={active_conda!r}, esperado={expected_conda!r}")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": active_conda,
        "gpu": [line for line in gpu.stdout.splitlines() if line.strip()],
    }


def _resolve_root(spec: dict[str, Any]) -> Path:
    return Path(spec["remote_root"]).expanduser().resolve()


def _collect_artifacts(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    experiment_dir = root / spec["experiment_directory"]
    run_id = spec["run_id"]
    contract = spec["artifact_contract"]
    found: dict[str, Any] = {"checkpoints": [], "terminal_receipt": None}
    terminal = experiment_dir / contract["terminal_receipt"].format(run_id=run_id)
    if not terminal.is_file():
        raise ExperimentError(f"falta recibo terminal del launcher: {terminal}")
    found["terminal_receipt"] = {
        "path": str(terminal.relative_to(root)),
        "sha256": sha256_file(terminal),
    }
    for pattern in contract["checkpoints"]:
        expanded = str(experiment_dir / pattern.format(run_id=run_id))
        for raw_path in sorted(glob.glob(expanded)):
            path = Path(raw_path)
            if path.is_file():
                found["checkpoints"].append(
                    {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                )
    minimum = int(contract.get("minimum_checkpoints", 1))
    if len(found["checkpoints"]) < minimum:
        raise ExperimentError(f"se esperaban al menos {minimum} checkpoints, se encontraron {len(found['checkpoints'])}")
    maximum = int(contract.get("maximum_checkpoints", minimum))
    if len(found["checkpoints"]) > maximum:
        raise ExperimentError(f"se esperaban como máximo {maximum} checkpoints, se encontraron {len(found['checkpoints'])}")
    return found


def run(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = _resolve_root(spec)
    experiment_dir = root / spec["experiment_directory"]
    run_dir = root / spec["run_directory"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec['run_id']}.log"
    receipt_path = run_dir / "RECEIPT.json"
    heartbeat_path = run_dir / "HEARTBEAT.json"
    lock_path = experiment_dir / "artifacts" / ".worker.lock"
    environment: dict[str, Any] = {}
    started_at = utc_now()

    with exclusive_lock(lock_path, nonblocking=True):
        try:
            environment = _environment(spec)
            free_bytes = shutil.disk_usage(experiment_dir).free
            if free_bytes < 5 * 1024**3:
                raise ExperimentError(f"espacio libre insuficiente: {free_bytes} bytes")
            data_dir = root / "data" / "train"
            if not data_dir.is_dir():
                raise ExperimentError(f"dataset remoto ausente: {data_dir}")

            entrypoint = experiment_dir / spec["entrypoint"]
            if not entrypoint.is_file():
                raise ExperimentError(f"entrypoint remoto ausente: {entrypoint}")
            command = [
                sys.executable,
                str(entrypoint),
                *spec.get("arguments", []),
                "--project-root",
                str(root),
                "--run-dir",
                str(run_dir),
                "--compact",
            ]
            running = {
                "schema_version": 1,
                "run_id": spec["run_id"],
                "experiment_id": spec["experiment_id"],
                "status": "RUNNING",
                "started_at": started_at,
                "worker_pid": os.getpid(),
                "command": command,
                "environment": environment,
                "config_sha256": spec["config_sha256"],
                "validation": spec["data_policy"].get("validation", "unknown"),
                "artifacts": {},
                "metrics": {},
            }
            atomic_write_json(receipt_path, running)
            atomic_write_json(heartbeat_path, {"status": "RUNNING", "at": utc_now(), "pid": os.getpid()})
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"event": "worker-start", "at": started_at, "command": command}) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
                running["child_pid"] = process.pid
                atomic_write_json(receipt_path, running)
                interval = max(5, int(spec.get("heartbeat_seconds", 30)))
                while True:
                    exit_code = process.poll()
                    atomic_write_json(
                        heartbeat_path,
                        {"status": "RUNNING" if exit_code is None else "EXITED", "at": utc_now(), "pid": process.pid},
                    )
                    if exit_code is not None:
                        break
                    time.sleep(interval)

            if exit_code != 0:
                raise ExperimentError(f"launcher terminó con código {exit_code}; ver {log_path}")
            artifacts = _collect_artifacts(spec, root)
            launcher_receipt_path = root / artifacts["terminal_receipt"]["path"]
            launcher_receipt = json.loads(launcher_receipt_path.read_text(encoding="utf-8"))
            final = {
                **running,
                "status": "SUCCEEDED",
                "completed_at": utc_now(),
                "exit_code": 0,
                "artifacts": artifacts,
                "metrics": launcher_receipt.get("metrics", {}),
            }
            atomic_write_json(receipt_path, final)
            atomic_write_json(heartbeat_path, {"status": "SUCCEEDED", "at": utc_now(), "pid": process.pid})
            return final
        except Exception as error:
            failure = {
                "schema_version": 1,
                "run_id": spec.get("run_id"),
                "experiment_id": spec.get("experiment_id"),
                "status": "FAILED",
                "started_at": started_at,
                "failed_at": utc_now(),
                "exit_code": locals().get("exit_code", 1),
                "environment": environment,
                "config_sha256": spec.get("config_sha256"),
                "validation": spec.get("data_policy", {}).get("validation", "unknown"),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "artifacts": {},
                "metrics": {},
            }
            atomic_write_json(receipt_path, failure)
            atomic_write_json(heartbeat_path, {"status": "FAILED", "at": utc_now(), "pid": os.getpid()})
            return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.spec.expanduser().resolve())
    print(json.dumps({"run_id": result.get("run_id"), "status": result["status"]}, sort_keys=True))
    raise SystemExit(0 if result["status"] == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
