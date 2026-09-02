#!/usr/bin/env python3
"""Opt-in scheduler for READY Biohub experiments.

Only manifests with ``execution_policy.auto_launch=true`` are eligible. This
keeps legacy/current experiments manual while allowing future variants to
consume the best available remote GPU without an agent remembering each step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import (
    ExperimentError,
    atomic_write_json,
    exclusive_lock,
    load_manifest,
    load_queue,
    read_json,
    refresh_queue,
    repo_root,
    utc_now,
)


def eligible_items(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repo_root()
    queue = refresh_queue(root)
    eligible: list[dict[str, Any]] = []
    for item in queue["items"]:
        if item.get("status") != "READY":
            continue
        _, manifest = load_manifest(item["experiment_id"], root)
        if manifest.get("execution_policy", {}).get("auto_launch") is True:
            eligible.append(item)
    return sorted(eligible, key=lambda row: (row.get("priority", 100), row["experiment_id"]))


def run_once(*, dry_run: bool = False, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    items = eligible_items(root)
    queue = load_queue(root)
    policy = read_json(root / ".agents" / "autonomous-exploration.json", {})
    max_concurrent = int(policy.get("budgets", {}).get("max_concurrent_remote_runs", 2))
    running = [item["experiment_id"] for item in queue["items"] if item.get("status") == "RUNNING"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "eligible": [item["experiment_id"] for item in items],
        "launched": None,
        "dry_run": dry_run,
        "running": running,
        "max_concurrent_remote_runs": max_concurrent,
    }
    if len(running) >= max_concurrent:
        result["blocked_reason"] = "global remote concurrency budget reached"
        return result
    if not items:
        return result
    experiment_id = items[0]["experiment_id"]
    command = [sys.executable, str(root / "infra" / "experimentctl.py"), "run", experiment_id, "--node", "auto"]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    result["returncode"] = completed.returncode
    result["controller"] = (completed.stdout or completed.stderr).strip().splitlines()[-1:]
    if completed.returncode == 0 and not dry_run:
        result["launched"] = experiment_id
    return result


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one scheduling cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="probe/select but never launch")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = repo_root()
    state_path = root / ".state" / "scheduler.json"
    try:
        with exclusive_lock(root / ".state" / "scheduler.lock", nonblocking=True):
            while True:
                result = run_once(dry_run=args.dry_run, root=root)
                atomic_write_json(state_path, result)
                print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
                if args.once:
                    return
                time.sleep(max(15, args.interval))
    except ExperimentError as error:
        print(json.dumps({"ok": False, "action": "scheduler", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
