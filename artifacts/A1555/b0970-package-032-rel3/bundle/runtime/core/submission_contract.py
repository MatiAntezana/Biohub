"""Structural submission checks independent of annotations or a scoring proxy."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path

COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def validate_csv(path: Path, *, enforce_row_budget: bool = True, expected_datasets=None):
    nodes = {}
    edges = set()
    datasets = set()
    incoming, outgoing = Counter(), Counter()
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected submission columns")
        for row in reader:
            if int(row["id"]) != rows:
                raise ValueError("CSV row IDs must be unique, contiguous, and zero-based")
            rows += 1
            dataset = row["dataset"]
            if not dataset or "/" in dataset or "\\" in dataset:
                raise ValueError("Invalid dataset identifier")
            datasets.add(dataset)
            if row["row_type"] == "node":
                key = (dataset, int(row["node_id"]))
                if key in nodes or key[1] < 0:
                    raise ValueError("Duplicate or negative node ID")
                frame = int(row["t"])
                if frame < 0 or any(not math.isfinite(float(row[a])) or float(row[a]) < 0 for a in ("z", "y", "x")):
                    raise ValueError("Invalid node frame/coordinates")
                if int(row["source_id"]) != -1 or int(row["target_id"]) != -1:
                    raise ValueError("Node row has edge endpoints")
                nodes[key] = frame
            elif row["row_type"] == "edge":
                edge = (dataset, int(row["source_id"]), int(row["target_id"]))
                if edge in edges:
                    raise ValueError("Duplicate edge")
                if any(float(row[key]) != -1 for key in ("node_id", "t", "z", "y", "x")):
                    raise ValueError("Edge row has node fields")
                edges.add(edge)
            else:
                raise ValueError("Unexpected row_type")
    for dataset, source, target in edges:
        u, v = (dataset, source), (dataset, target)
        if u not in nodes or v not in nodes:
            raise ValueError("Dangling edge")
        if nodes[v] != nodes[u] + 1:
            raise ValueError("Non-consecutive or backwards edge")
        incoming[v] += 1
        outgoing[u] += 1
        if incoming[v] > 1 or outgoing[u] > 2:
            raise ValueError("Invalid lineage degree: at most one parent and two children")
    if not nodes or not edges:
        raise ValueError("Empty predicted lineage")
    if enforce_row_budget and not 234000 < rows < 242000:
        raise ValueError(f"Submission outside user row budget: {rows}")
    if expected_datasets is not None and datasets != set(expected_datasets):
        raise ValueError("Submission datasets differ from expected test samples")
    return {"csv_rows": rows, "nodes": len(nodes), "edges": len(edges),
            "divisions": sum(count == 2 for count in outgoing.values()),
            "datasets": sorted(datasets), "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "valid": True}
