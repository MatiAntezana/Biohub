"""Official scorer adapter. Imported only after label-free prediction."""
from pathlib import Path
import sys
from core.artifacts import ROOT


def make_graph(nodes_by_t, edges):
    import polars as pl
    import tracksdata as td
    graph = td.graph.InMemoryGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Float64, 0.0)
    ids = {}
    for t, nodes in sorted(nodes_by_t.items()):
        for node in nodes:
            if node["id"] in ids:
                raise ValueError("Duplicate graph ID")
            ids[node["id"]] = graph.add_node({"t": int(t), **dict(zip(("z", "y", "x"), map(float, node["pos"])))})
    for u, v in edges:
        graph.add_edge(ids[u], ids[v], {})
    return graph


def metric_root():
    return ROOT / "vendor/official_metric/src"
