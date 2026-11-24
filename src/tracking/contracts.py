"""Input identity, lineage invariants and explicit remote-only execution."""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import socket
import sys

import numpy as np

PRIMARY_SHA = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
SECONDARY_SHA = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing explicit SHA-256 for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def require_worker(node_name):
    from core.remote_executor import DEFAULT_NODES, validate_node
    node = DEFAULT_NODES[node_name]
    validate_node(node)
    if platform.system() != "Linux" or socket.gethostname().split(".")[0].upper() not in {node.name, f"{node.name}-UBU"}:
        raise RuntimeError("--local is only valid on the selected UCBV Linux worker, never on the Mac")
    if Path(sys.executable).resolve() != Path(node.python_path).resolve():
        raise RuntimeError(f"Use the configured interpreter: {node.python_path}")


@dataclass
class Lineage:
    # Stable local IDs are row indices; coordinates are pixels in t,z,y,x order.
    coords: np.ndarray
    edges: np.ndarray

    def validate(self, shape=None):
        c, e = np.asarray(self.coords), np.asarray(self.edges)
        if c.ndim != 2 or c.shape[1] != 4 or not len(c) or not np.isfinite(c).all():
            raise ValueError("Expected nonempty finite N x 4 tzyx coordinates")
        if (c < 0).any() or not np.equal(c[:, 0], np.floor(c[:, 0])).all():
            raise ValueError("Negative coordinates or fractional frame")
        if len({tuple(row) for row in c}) != len(c):
            raise ValueError("Duplicate detections")
        if shape is not None and (c >= np.asarray(shape)).any():
            raise ValueError("Coordinates outside the image")
        if e.ndim != 2 or e.shape[1] != 2 or not np.issubdtype(e.dtype, np.integer):
            raise ValueError("Edges must be an integer E x 2 array")
        if len(e):
            if (e < 0).any() or (e >= len(c)).any() or len(set(map(tuple, e))) != len(e):
                raise ValueError("Invalid or duplicate endpoints")
            if not (c[e[:, 1], 0] == c[e[:, 0], 0] + 1).all():
                raise ValueError("Only consecutive forward edges are allowed")
            if max(Counter(e[:, 1]).values()) > 1 or max(Counter(e[:, 0]).values()) > 2:
                raise ValueError("At most one parent and two daughters per node")
        return self


def load_reference(path, names):
    from core.submission_contract import validate_csv
    validate_csv(Path(path), enforce_row_budget=False)
    nodes, edges = {n: {} for n in names}, {n: [] for n in names}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["dataset"]
            if name not in nodes:
                continue
            if row["row_type"] == "node":
                nodes[name][int(row["node_id"])] = [float(row[k]) for k in ("t", "z", "y", "x")]
            else:
                edges[name].append((int(row["source_id"]), int(row["target_id"])))
    result = {}
    for name in names:
        ids = sorted(nodes[name])
        mapping = {old: new for new, old in enumerate(ids)}
        result[name] = Lineage(np.asarray([nodes[name][i] for i in ids], dtype=float).reshape(-1, 4),
            np.asarray([(mapping[u], mapping[v]) for u, v in edges[name]], dtype=np.int64).reshape(-1, 2)).validate()
    return result


def check_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("scope") not in {"diagnostic", "submission"}:
        raise ValueError("Unknown manifest schema/scope")
    rows = manifest["datasets"]
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicated dataset panel")
    groups = {}
    for row in rows:
        if not isinstance(row["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["name"]):
            raise ValueError("Dataset name must be a safe identifier, not a path")
        if not row.get("group") or row["split"] not in {"train", "validation", "confirmation", "test"}:
            raise ValueError("Explicit biological group and split required")
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Biological group crosses splits: do not randomly split frames/crops")
        spacing = np.asarray(row["scale_um"], dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Explicit positive z,y,x spacing in micrometres required")
    if manifest["scope"] == "submission":
        if any(r["split"] != "test" or "gt" in r for r in rows):
            raise ValueError("Submission manifests must be test-only and contain no GT paths")


def load_candidates(row):
    verify_file(row["candidates"], row["candidates_sha256"])
    receipt = json.loads(Path(row["candidates_receipt"]).read_text())
    if (receipt.get("dataset") != row["name"] or receipt.get("primary_sha256") != PRIMARY_SHA
            or receipt.get("secondary_sha256") != SECONDARY_SHA):
        raise ValueError("Cache is not the identified production dual-seed ensemble")
    # Dense caches preserve weak alternatives; do not silently accept the old fast-loop cache.
    if not 0 <= receipt.get("candidate_threshold", 1) <= .05:
        raise ValueError("Regenerate the dual-seed cache with candidate threshold <= 0.05")
    with np.load(row["candidates"], allow_pickle=False) as data:
        coords, edges = np.asarray(data["coords"], float), np.asarray(data["edges"], float)
    if edges.size == 0:
        edges = np.empty((0, 4), dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("Raw candidates must have columns source,target,probability,distance")
    Lineage(coords, np.empty((0, 2), dtype=np.int64)).validate()
    if not np.isfinite(edges).all() or (edges[:, 2:] < 0).any() or (edges[:, 2] > 1).any():
        raise ValueError("Invalid candidate probabilities/distances")
    uv = edges[:, :2]
    if (uv < 0).any() or (uv >= len(coords)).any() or not np.equal(uv, np.floor(uv)).all():
        raise ValueError("Invalid raw candidate node IDs")
    return coords, edges


def write_submission(path, predictions, *, enforce_budget):
    from core.submission_contract import COLUMNS, validate_csv
    with Path(path).open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        row_id = 0
        for name, graph in sorted(predictions.items()):
            graph.validate()
            for node_id, (t, z, y, x) in enumerate(graph.coords):
                writer.writerow([row_id, name, "node", node_id, int(t), z, y, x, -1, -1])
                row_id += 1
            for u, v in sorted(map(tuple, graph.edges)):
                writer.writerow([row_id, name, "edge", -1, -1, -1, -1, -1, int(u), int(v)])
                row_id += 1
    return validate_csv(Path(path), enforce_row_budget=enforce_budget, expected_datasets=predictions)
