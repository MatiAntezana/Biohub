"""
Remote Execution Engine for Autoresearch V2.
Enforces that ALL evaluation and model experimentation runs strictly on authorized
remote GPU cluster nodes (A1552, A1556, A1557, A1560), while the Mac acts
purely as the Controller / Supervisor.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "cluster_config.json"


@dataclass
class RemoteNode:
    name: str
    ip: str
    port: int = 2222
    user: str = "alumno1"
    conda_env: str = "train-56"
    python_path: str = "/home/alumno1/miniconda3/envs/train-56/bin/python"
    remote_workdir: str = "/home/alumno1/Documents/Biohub-mejorado"


DEFAULT_NODES = {
    "A1556-UBU": RemoteNode("A1556-UBU", "100.107.113.22", conda_env="train-56", python_path="/home/alumno1/miniconda3/envs/train-56/bin/python"),
    "A1552-UBU": RemoteNode("A1552-UBU", "100.112.138.117", conda_env="train-52", python_path="/home/alumno1/miniconda3/envs/train-52/bin/python"),
    "A1560-UBU": RemoteNode("A1560-UBU", "100.84.253.117", conda_env="train-60", python_path="/home/alumno1/miniconda3/envs/train-60/bin/python"),
    "A1557-UBU": RemoteNode("A1557-UBU", "100.87.128.33", conda_env="train-57", python_path="/home/alumno1/miniconda3/envs/train-57/bin/python"),
}


def load_cluster_config() -> Dict[str, Any]:
    """Loads or creates default cluster configuration."""
    if not CONFIG_PATH.exists():
        default_config = {
            "enforce_remote_execution": True,
            "default_node": "A1556-UBU",
            "sync_code_before_run": True,
            "ssh_port": 2222,
            "ssh_user": "alumno1",
            "remote_workdir": "/home/alumno1/Documents/Biohub-mejorado",
            "allowed_nodes": list(DEFAULT_NODES.keys()),
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_online_node(preferred: Optional[str] = None) -> RemoteNode:
    """Finds an online cluster node, prioritizing `preferred` or max free VRAM."""
    config = load_cluster_config()
    target_name = preferred or config.get("default_node", "A1556-UBU")

    # First check preferred
    if target_name in DEFAULT_NODES:
        node = DEFAULT_NODES[target_name]
        if check_node_online(node):
            return node

    # Fallback to scanning others
    for name, node in DEFAULT_NODES.items():
        if name != target_name and check_node_online(node):
            print(f"[REMOTE] Target {target_name} is offline. Falling back to online node: {name}")
            return node

    raise ConnectionError(
        "No remote cluster nodes are reachable via SSH over Tailscale. "
        "Verify that Tailscale is connected and at least one node is online."
    )


def check_node_online(node: RemoteNode, timeout: int = 4) -> bool:
    """Quickly tests if a remote node is reachable."""
    cmd = [
        "ssh", "-p", str(node.port),
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        f"{node.user}@{node.ip}",
        "hostname",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
        return res.returncode == 0
    except Exception:
        return False


def sync_code_to_remote(node: RemoteNode, quiet: bool = False) -> bool:
    """Syncs the Biohub-mejorado codebase from Mac controller to remote machine via rsync."""
    if not quiet:
        print(f"[REMOTE] Syncing code to {node.name} ({node.ip}:{node.remote_workdir})...")

    # Ensure remote directory exists
    mkdir_cmd = [
        "ssh", "-p", str(node.port),
        "-o", "BatchMode=yes",
        f"{node.user}@{node.ip}",
        f"mkdir -p {node.remote_workdir}",
    ]
    subprocess.run(mkdir_cmd, capture_output=True, check=True)

    rsync_cmd = [
        "rsync", "-avz",
        "--delete",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".DS_Store",
        "-e", f"ssh -p {node.port} -o BatchMode=yes",
        f"{PROJECT_ROOT}/",
        f"{node.user}@{node.ip}:{node.remote_workdir}/",
    ]
    res = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[REMOTE] Rsync failed: {res.stderr}")
        return False
    if not quiet:
        print(f"[REMOTE] Code synced successfully.")
    return True


def run_remote_python(
    script_relative_path: str,
    args: List[str],
    node: Optional[RemoteNode] = None,
    sync_first: bool = True,
    quiet: bool = False,
) -> Tuple[int, str, str]:
    """
    Executes a python script remotely on the cluster node.
    Returns: (returncode, stdout, stderr)
    """
    target_node = node or find_best_online_node()

    if sync_first:
        sync_code_to_remote(target_node, quiet=quiet)

    remote_script = f"{target_node.remote_workdir}/{script_relative_path}"
    cmd_str = f"cd {target_node.remote_workdir} && {target_node.python_path} {remote_script} {' '.join(args)}"

    if not quiet:
        print(f"[REMOTE] Executing on {target_node.name} via {target_node.python_path}...")

    ssh_cmd = [
        "ssh", "-p", str(target_node.port),
        "-o", "BatchMode=yes",
        f"{target_node.user}@{target_node.ip}",
        cmd_str,
    ]

    proc = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


if __name__ == "__main__":
    print("Verifying Remote Execution Engine...")
    node = find_best_online_node()
    print(f"Selected node: {node.name} ({node.ip})")
    sync_code_to_remote(node)
    code, out, err = run_remote_python("fast_loop/evaluate_fast.py", ["--quiet"], node=node)
    print("Remote execution result code:", code)
    if code == 0:
        print("Remote test succeeded!")
    else:
        print("Remote test error:", err)
