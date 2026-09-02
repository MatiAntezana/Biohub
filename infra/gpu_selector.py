#!/usr/bin/env python3
"""Select a remote Biohub GPU by live free VRAM.

The selector is intentionally side-effect free: it only probes registered
nodes. Launching remains the responsibility of ``experimentctl run``.
"""

from __future__ import annotations

import csv
import io
import shlex
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Callable

from experimentlib import ExperimentError


@dataclass(frozen=True)
class GPUCandidate:
    node: str
    gpu_index: int
    free_mib: int
    total_mib: int
    utilization_pct: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ssh_command(node: dict[str, Any], remote_command: str) -> list[str]:
    host = str(node.get("host", "")).strip()
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]
    # If host is an SSH config alias like "a1552", "a1555", etc.
    if host.lower().startswith("a1") or ("." not in host and ":" not in host):
        cmd.extend([host.lower(), "bash", "-lc", shlex.quote(remote_command)])
    else:
        cmd.extend(["-p", str(node.get("port", 2222)), f"{node['user']}@{host}", "bash", "-lc", shlex.quote(remote_command)])
    return cmd


def parse_nvidia_smi(text: str, node_name: str) -> list[GPUCandidate]:
    candidates: list[GPUCandidate] = []
    for row in csv.reader(io.StringIO(text)):
        if not row or len(row) < 4:
            continue
        try:
            index, free_mib, total_mib, utilization = (int(value.strip()) for value in row[:4])
        except ValueError:
            continue
        candidates.append(
            GPUCandidate(
                node=node_name,
                gpu_index=index,
                free_mib=free_mib,
                total_mib=total_mib,
                utilization_pct=utilization,
            )
        )
    return candidates


def probe_node_gpus(
    node_name: str,
    node: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[list[GPUCandidate], str | None]:
    if node.get("status") != "ONLINE" or node.get("training_ready") is not True:
        return [], f"status={node.get('status', 'UNKNOWN')}"
    command = (
        "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    result = runner(_ssh_command(node, command), capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "probe failed").strip()[-300:]
        return [], detail
    candidates = parse_nvidia_smi(result.stdout, node_name)
    return candidates, None if candidates else "nvidia-smi returned no GPUs"


def select_gpu(
    nodes: dict[str, dict[str, Any]],
    allowed_nodes: list[str],
    *,
    min_free_vram_mib: int = 0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[GPUCandidate, dict[str, Any]]:
    candidates: list[GPUCandidate] = []
    probes: dict[str, Any] = {}
    for node_name in sorted(allowed_nodes):
        node = nodes.get(node_name)
        if node is None:
            probes[node_name] = {"error": "unregistered node", "gpus": []}
            continue
        found, error = probe_node_gpus(node_name, node, runner=runner)
        probes[node_name] = {"error": error, "gpus": [item.as_dict() for item in found]}
        candidates.extend(item for item in found if item.free_mib >= min_free_vram_mib)
    if not candidates:
        raise ExperimentError(
            f"ninguna GPU permitida tiene al menos {min_free_vram_mib} MiB libres; probes={probes}"
        )
    # Most free memory wins. Lower utilization and stable node/index order break ties.
    selected = sorted(
        candidates,
        key=lambda item: (-item.free_mib, item.utilization_pct, item.node, item.gpu_index),
    )[0]
    return selected, probes
