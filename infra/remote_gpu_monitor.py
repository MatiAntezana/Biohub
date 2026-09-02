#!/usr/bin/env python3
"""Read-only discovery of experiment processes on enabled remote GPUs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from aienv import ensure_aienv
from experimentlib import atomic_write_json, load_nodes, load_queue, read_json, repo_root, utc_now


EXPERIMENT_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])exp[-_ ]?(\d{3}[a-z]?)(?:[^a-z0-9]|$)")


def infer_experiment_id(command: str) -> str | None:
    match = EXPERIMENT_PATTERN.search(command)
    return f"EXP-{match.group(1).upper()}" if match else None


def _ssh_command(node: dict[str, Any], remote_code: str) -> list[str]:
    inner = f"python3 -c {shlex.quote(remote_code)}"
    return [
        "ssh",
        "-p",
        str(node["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        f"{node['user']}@{node['host']}",
        "bash",
        "-lc",
        shlex.quote(inner),
    ]


REMOTE_PROBE = r'''
import json, subprocess

def call(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr

gpu_rc, gpu_out, gpu_err = call([
    "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits"
])
app_rc, app_out, app_err = call([
    "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits"
])
commands = {}
for raw in app_out.splitlines():
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 2 or not parts[1].isdigit():
        continue
    pid = parts[1]
    ps = subprocess.run(["ps", "-p", pid, "-o", "args="], capture_output=True, text=True, check=False)
    commands[pid] = ps.stdout.strip()
print(json.dumps({
    "gpu_rc": gpu_rc, "gpu_out": gpu_out, "gpu_error": gpu_err.strip(),
    "app_rc": app_rc, "app_out": app_out, "app_error": app_err.strip(),
    "commands": commands,
}, sort_keys=True))
'''


def _integer(value: str) -> int | None:
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def parse_remote_payload(node_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    for raw in str(payload.get("gpu_out", "")).splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 7:
            continue
        index = _integer(parts[0])
        if index is None:
            continue
        row = {
            "index": index,
            "uuid": parts[1],
            "name": parts[2],
            "total_mib": _integer(parts[3]),
            "used_mib": _integer(parts[4]),
            "free_mib": _integer(parts[5]),
            "utilization_pct": _integer(parts[6]),
        }
        gpus.append(row)
        uuid_to_index[parts[1]] = index

    activities: list[dict[str, Any]] = []
    commands = payload.get("commands", {}) if isinstance(payload.get("commands"), dict) else {}
    for raw in str(payload.get("app_out", "")).splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 4:
            continue
        pid = _integer(parts[1])
        if pid is None:
            continue
        command = str(commands.get(str(pid), ""))[:800]
        gpu_uuid = parts[0]
        experiment_id = infer_experiment_id(command)
        activity = {
            "activity_id": f"{node_name}:{gpu_uuid}:{pid}",
            "node": node_name,
            "gpu_index": uuid_to_index.get(gpu_uuid),
            "gpu_uuid": gpu_uuid,
            "pid": pid,
            "process_name": parts[2],
            "used_gpu_memory_mib": _integer(parts[3]),
            "experiment_id": experiment_id,
            "association": "command" if experiment_id else "unknown",
            "command": command,
        }
        activities.append(activity)
    error = None
    if int(payload.get("gpu_rc", 0) or 0) != 0:
        error = str(payload.get("gpu_error") or "nvidia-smi GPU query failed")[-500:]
    elif int(payload.get("app_rc", 0) or 0) != 0:
        error = str(payload.get("app_error") or "nvidia-smi process query failed")[-500:]
    return {"node": node_name, "gpus": gpus, "activities": activities, "error": error}


def probe_node_activity(
    node_name: str,
    node: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if node.get("status") != "ONLINE" or node.get("training_ready") is not True:
        return {"node": node_name, "gpus": [], "activities": [], "error": f"status={node.get('status', 'UNKNOWN')}"}
    result = runner(_ssh_command(node, REMOTE_PROBE), capture_output=True, text=True, check=False)
    if result.returncode:
        return {
            "node": node_name,
            "gpus": [],
            "activities": [],
            "error": (result.stderr or result.stdout or "SSH probe failed").strip()[-500:],
        }
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        return {"node": node_name, "gpus": [], "activities": [], "error": f"invalid probe JSON: {error}"}
    return parse_remote_payload(node_name, payload)


def scan_remote_gpus(
    nodes: dict[str, dict[str, Any]],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    enabled = {
        name: node
        for name, node in nodes.items()
        if node.get("status") == "ONLINE" and node.get("training_ready") is True
    }
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(enabled)))) as pool:
        futures = {pool.submit(probe_node_activity, name, node, runner=runner): name for name, node in enabled.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = {"node": name, "gpus": [], "activities": [], "error": repr(error)}
    activities = [activity for name in sorted(results) for activity in results[name]["activities"]]
    return {"schema_version": 1, "scanned_at": utc_now(), "nodes": results, "activities": activities}


def _transitions(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    before = {row["activity_id"]: row for row in previous.get("activities", [])}
    after = {row["activity_id"]: row for row in current.get("activities", [])}
    transitions: list[dict[str, Any]] = []
    for activity_id in sorted(after.keys() - before.keys()):
        transitions.append({"type": "GPU_PROCESS_STARTED", **after[activity_id]})
    for activity_id in sorted(before.keys() - after.keys()):
        transitions.append({"type": "GPU_PROCESS_STOPPED", **before[activity_id]})
    for activity_id in sorted(before.keys() & after.keys()):
        comparable = ("experiment_id", "gpu_index", "used_gpu_memory_mib", "process_name")
        if any(before[activity_id].get(key) != after[activity_id].get(key) for key in comparable):
            transitions.append({"type": "GPU_PROCESS_CHANGED", "before": before[activity_id], "after": after[activity_id]})
    return transitions


def scan_and_store(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    path = root / ".state" / "remote-gpu-activity.json"
    previous = read_json(path, {"activities": []})
    current = scan_remote_gpus(load_nodes(root))
    failed_nodes = {name for name, row in current.get("nodes", {}).items() if row.get("error")}
    current_ids = {row["activity_id"] for row in current["activities"]}
    for activity in previous.get("activities", []):
        if activity.get("node") in failed_nodes and activity.get("activity_id") not in current_ids:
            preserved = {
                **activity,
                "stale": True,
                "probe_error": current["nodes"][activity["node"]]["error"],
                "last_seen_at": previous.get("scanned_at"),
            }
            current["activities"].append(preserved)
    registered = {
        (item.get("experiment_id"), item.get("node") or item.get("requested_node")): item
        for item in load_queue(root).get("items", [])
        if item.get("status") == "RUNNING"
    }
    for activity in current["activities"]:
        item = registered.get((activity.get("experiment_id"), activity.get("node")))
        if item:
            activity["association"] = "registered"
            activity["run_id"] = item.get("run_id")
    current["transitions"] = _transitions(previous, current)
    atomic_write_json(path, current)
    return current


def compact_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scanned_at": payload.get("scanned_at"),
        "experiments_running": sorted({row["experiment_id"] for row in payload.get("activities", []) if row.get("experiment_id")}),
        "activities": [
            {
                "node": row.get("node"),
                "gpu_index": row.get("gpu_index"),
                "pid": row.get("pid"),
                "experiment_id": row.get("experiment_id"),
                "used_gpu_memory_mib": row.get("used_gpu_memory_mib"),
                "association": row.get("association"),
                "run_id": row.get("run_id"),
                "stale": row.get("stale", False),
                "probe_error": row.get("probe_error"),
            }
            for row in payload.get("activities", [])
        ],
        "transitions": payload.get("transitions", []),
        "node_errors": {name: row["error"] for name, row in payload.get("nodes", {}).items() if row.get("error")},
    }


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "status"), default="scan", nargs="?")
    args = parser.parse_args()
    root = repo_root()
    if args.command == "scan":
        payload = scan_and_store(root)
    else:
        payload = read_json(root / ".state" / "remote-gpu-activity.json", {"activities": [], "nodes": {}})
    print(json.dumps({"ok": True, "action": f"remote-gpu-{args.command}", **compact_view(payload)}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
