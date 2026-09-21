"""UCBV alias-only dispatch to immutable code releases; no automatic node switch."""
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
import tempfile

from core.artifacts import ROOT, code_hash, code_manifest

CONFIG = json.loads((ROOT / "configs/cluster.json").read_text())
LOCAL_PYTHON = "/Users/matias/miniforge3/envs/AIEnv/bin/python"


@dataclass(frozen=True)
class RemoteNode:
    name: str
    ssh_alias: str
    python_path: str


DEFAULT_NODES = {name: RemoteNode(name, row["alias"],
    f"/home/alumno1/miniconda3/envs/{row['environment']}/bin/python")
    for name, row in CONFIG["nodes"].items()}


def validate_node(node):
    if node.name not in DEFAULT_NODES or node != DEFAULT_NODES[node.name]:
        raise ValueError("Unknown or changed node identity; no direct IP/host-key fallback")


def require_worker(node):
    validate_node(node)
    if platform.system() != "Linux" or socket.gethostname().upper().split(".")[0] not in {node.name, node.name + "-UBU"}:
        raise RuntimeError("Worker action requires the selected UCBV Linux host")
    if Path(sys.executable).resolve() != Path(node.python_path).resolve():
        raise RuntimeError("Wrong remote conda interpreter")


def execute_remotely(file_path, node, arguments):
    """Sync only code to a hash-named release and invoke a short worker action.

    Long jobs are spawned by the worker and write persistent receipts. Status
    actions return promptly. The code, runs, assets and pre-existing data have
    distinct roots. rsync never uses --delete and never sends old run receipts.
    """
    validate_node(node)
    if platform.system() != "Darwin" or Path(sys.executable).resolve() != Path(LOCAL_PYTHON).resolve():
        raise RuntimeError("Dispatch must run from the Mac with AIEnv")
    relative = Path(file_path).resolve().relative_to(ROOT)
    identity = code_hash()
    release = CONFIG["remote_root"] + "/releases/" + identity
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15"]
    subprocess.run([*ssh, node.ssh_alias, "mkdir -p " + shlex.quote(release)], check=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix="biohub-code-") as listing:
        listing.write("\n".join(code_manifest()) + "\n")
        listing.flush()
        subprocess.run(["rsync", "-az", "--files-from=" + listing.name,
            "-e", shlex.join(ssh), str(ROOT) + "/", node.ssh_alias + ":" + release + "/"], check=True)
    command = [node.python_path, release + "/" + str(relative), *arguments,
        "--worker", "--node", node.name, "--artifact-root", CONFIG["remote_root"], "--code-hash", identity]
    result = subprocess.run([*ssh, node.ssh_alias, "cd " + shlex.quote(release) + " && " + shlex.join(command)],
        text=True, capture_output=True, timeout=60)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    return True
