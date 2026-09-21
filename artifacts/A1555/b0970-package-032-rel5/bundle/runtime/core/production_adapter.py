"""Load the frozen production graph implementation without running inference.

This adapter deliberately uses the notebook's actual configuration and functions
so offline ablations and production don't evolve as separate implementations.
The snapshot is trusted project code, never an arbitrary uploaded notebook.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path


def load_production_namespace(notebook: Path) -> dict:
    nb = json.loads(notebook.read_text())
    namespace = {"__name__": "production_graph_audit"}
    with contextlib.redirect_stdout(io.StringIO()):
        for index in (0, 2):
            tree = ast.parse("".join(nb["cells"][index]["source"]))
            # display() is a notebook-only dependency; it is never called by
            # the configuration or graph functions used in this adapter.
            tree.body = [statement for statement in tree.body if not (
                isinstance(statement, ast.ImportFrom) and statement.module == "IPython.display"
            )]
            exec(compile(tree, f"{notebook}:cell{index}", "exec"), namespace)
    source = "".join(nb["cells"][5]["source"])
    tree = ast.parse(source)
    definitions = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEEPCENTER_VETO_DETECTOR"
            for target in statement.targets
        ):
            break
        definitions.append(statement)
    else:
        raise ValueError("Production graph boundary not found; refusing full notebook execution")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), f"{notebook}:graph", "exec"), namespace)
    return namespace


def graph_to_plain(graph):
    nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
              **{axis: float(r[axis]) for axis in ("z", "y", "x")}}
             for r in graph.node_attrs().iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
              "edge_prob": float(r["edge_prob"]) if r.get("edge_prob") is not None else None}
             for r in graph.edge_attrs().iter_rows(named=True)]
    return nodes, edges


def as_evaluator_inputs(nodes, edges, round_coordinates=False):
    by_t = {}
    for node_id, node in nodes.items():
        coords = [float(node[a]) for a in ("z", "y", "x")]
        if round_coordinates:
            coords = [float(max(0, round(c))) for c in coords]
        by_t.setdefault(node["t"], []).append({"id": node_id, "t": node["t"], "pos": coords})
    return by_t, [(e["source_id"], e["target_id"]) for e in edges]
