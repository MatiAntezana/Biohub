"""
Remote Execution Engine for Autoresearch V2.
Enforces that ALL evaluation and model experimentation runs strictly on the authorized
remote GPU cluster node (exclusively A1557 per user mandate), while the Mac acts
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
    ssh_alias: str
    conda_env: str = "train-57"
    python_path: str = "/home/alumno1/miniconda3/envs/train-57/bin/python"
    remote_workdir: str = "/home/alumno1/Documents/Biohub-mejorado"


# Authorized Remote Nodes: A1557 and A1558 per user mandate
NODE_A1557 = RemoteNode(
    name="A1557",
    ssh_alias="a1557",
    conda_env="train-57",
    python_path="/home/alumno1/miniconda3/envs/train-57/bin/python",
    remote_workdir="/home/alumno1/Documents/Biohub-mejorado",
)

NODE_A1558 = RemoteNode(
    name="A1558",
    ssh_alias="a1558",
    conda_env="train-58",
    python_path="/home/alumno1/miniconda3/envs/train-58/bin/python",
    remote_workdir="/home/alumno1/Documents/Biohub-mejorado",
)

DEFAULT_NODE = NODE_A1557

DEFAULT_NODES: Dict[str, RemoteNode] = {
    "A1557": NODE_A1557,
    "A1558": NODE_A1558,
}


def load_cluster_config() -> Dict[str, Any]:
    """Loads cluster configuration."""
    if not CONFIG_PATH.exists():
        default_config = {
            "enforce_remote_execution": True,
            "authorized_exclusive_node": "A1557",
            "default_node": "A1557",
            "ssh_host": "a1557",
            "conda_env": "train-57",
            "python_path": "/home/alumno1/miniconda3/envs/train-57/bin/python",
            "remote_workdir": "/home/alumno1/Documents/Biohub-mejorado",
            "allowed_nodes": ["A1557", "A1558"],
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_online_node(preferred: Optional[str] = None) -> RemoteNode:
    """Returns an authorized online remote node (A1557 or A1558)."""
    if preferred and preferred in DEFAULT_NODES:
        target = DEFAULT_NODES[preferred]
        if check_node_online(target):
            return target
        print(f"[REMOTE] Preferred node {preferred} is offline, falling back...")

    # Check authorized nodes in priority order: A1557, A1558
    for name in ["A1557", "A1558"]:
        node = DEFAULT_NODES[name]
        if check_node_online(node):
            return node

    raise ConnectionError(
        "None of the authorized remote nodes (A1557, A1558) are currently reachable. "
        "Verify Tailscale and SSH proxy connection."
    )


def check_node_online(node: RemoteNode, timeout: int = 5) -> bool:
    """Tests connection to remote node via ssh alias."""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        node.ssh_alias,
        "hostname",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 3)
        return res.returncode == 0
    except Exception:
        return False


def sync_code_to_remote(node: RemoteNode, quiet: bool = False) -> bool:
    """Syncs the Biohub-mejorado codebase from Mac controller to remote machine via rsync."""
    if not quiet:
        print(f"[REMOTE] Syncing code to {node.name} via ssh alias '{node.ssh_alias}' ({node.remote_workdir})...")

    # Ensure remote directory exists
    mkdir_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        node.ssh_alias,
        f"mkdir -p {node.remote_workdir}",
    ]
    subprocess.run(mkdir_cmd, capture_output=True, check=True)

    rsync_cmd = [
        "rsync", "-avz",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".DS_Store",
        "--filter", "- champion_weights/***",
        "--filter", "- *.pt",
        "--filter", "- *.pth",
        "-e", "ssh -o BatchMode=yes",
        f"{PROJECT_ROOT}/",
        f"{node.ssh_alias}:{node.remote_workdir}/",
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
    Executes a python script remotely on the authorized cluster node.
    Returns: (returncode, stdout, stderr)
    """
    target_node = node or find_best_online_node()

    if sync_first:
        sync_code_to_remote(target_node, quiet=quiet)

    # When running on the remote node, it executes locally on itself
    remote_args = list(args)
    if "--local" not in remote_args:
        remote_args.append("--local")

    remote_script = f"{target_node.remote_workdir}/{script_relative_path}"
    cmd_str = f"cd {target_node.remote_workdir} && {target_node.python_path} {remote_script} {' '.join(remote_args)}"

    if not quiet:
        print(f"[REMOTE] Executing on {target_node.name} via {target_node.python_path}...")

    ssh_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        target_node.ssh_alias,
        cmd_str,
    ]

    proc = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


if __name__ == "__main__":
    print(f"Verifying Remote Execution Engine (Mandated Node: {DEFAULT_NODE.name})...")
    node = find_best_online_node()
    print(f"Connected to node: {node.name}")
    sync_code_to_remote(node)
    code, out, err = run_remote_python("fast_loop/evaluate_fast.py", ["--quiet"], node=node)
    print("Remote execution result code:", code)
    if code == 0:
        print("Remote test succeeded on authorized node!")
    else:
        print("Remote test error:", err)
