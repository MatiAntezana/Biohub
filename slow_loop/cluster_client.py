"""
Remote GPU Cluster Client for Autoresearch V2 Slow-Loop.
Manages SSH communication with the 4x RTX 4090 cluster (A1552, A1556, A1557, A1560)
over Tailscale mesh on port 2222.
"""

from __future__ import annotations
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ClusterNode:
    name: str
    ip: str
    port: int = 2222
    user: str = "alumno1"
    gpu_name: str = "NVIDIA GeForce RTX 4090"
    vram_total_mib: int = 24576


CLUSTER_NODES: Dict[str, ClusterNode] = {
    "A1560-UBU": ClusterNode("A1560-UBU", "100.84.253.117"),
    "A1552-UBU": ClusterNode("A1552-UBU", "100.112.138.117"),
    "A1556-UBU": ClusterNode("A1556-UBU", "100.107.113.22"),
    "A1557-UBU": ClusterNode("A1557-UBU", "100.87.128.33"),
}


def run_remote_command(node: ClusterNode, command: str, timeout: int = 15) -> Tuple[int, str, str]:
    """Runs a shell command on a remote cluster node via SSH."""
    cmd = [
        "ssh",
        "-p", str(node.port),
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        f"{node.user}@{node.ip}",
        command,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Connection timed out"
    except Exception as e:
        return -1, "", str(e)


def query_node_gpu_status(node: ClusterNode) -> Dict[str, Any]:
    """Queries GPU memory usage and temperature via nvidia-smi."""
    cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
    code, stdout, stderr = run_remote_command(node, cmd)
    if code != 0:
        return {"online": False, "error": stderr.strip() or "Unreachable"}

    try:
        parts = [p.strip() for p in stdout.strip().split(",")]
        return {
            "online": True,
            "gpu_name": parts[0],
            "memory_total_mib": int(parts[1]),
            "memory_free_mib": int(parts[2]),
            "memory_used_mib": int(parts[3]),
            "gpu_utilization_pct": int(parts[4]),
            "temperature_c": int(parts[5]),
        }
    except Exception as e:
        return {"online": True, "parse_error": str(e), "raw": stdout}


def scan_cluster() -> Dict[str, Dict[str, Any]]:
    """Scans all 4 RTX 4090 nodes in parallel and returns status dict."""
    status = {}
    for name, node in CLUSTER_NODES.items():
        status[name] = query_node_gpu_status(node)
    return status


if __name__ == "__main__":
    print("Scanning Biohub GPU Cluster...")
    results = scan_cluster()
    for name, res in results.items():
        if res.get("online"):
            print(f"✅ {name}: {res.get('gpu_name')} | Free: {res.get('memory_free_mib')} MiB | Util: {res.get('gpu_utilization_pct')}% | Temp: {res.get('temperature_c')}°C")
        else:
            print(f"❌ {name}: OFFLINE ({res.get('error')})")
