"""
Cache Extractor for Autoresearch V2.
Extracts model detections and features from trained checkpoints or GEFF prediction files
and compiles them into standard `val_cache.json` for Fast-Loop evaluation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_geff_text_or_json(geff_path: Path) -> Dict[str, Any]:
    """
    Parses nodes and edges from a .geff file.
    GEFF files can be parsed as JSON or edge/node lists.
    """
    nodes_by_t: Dict[int, List[Dict[str, Any]]] = {}
    edges: List[Tuple[int, int]] = []

    try:
        with open(geff_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If standard json graph
        for node in data.get("nodes", []):
            t = int(node.get("t", 0))
            nodes_by_t.setdefault(t, []).append({
                "id": int(node["id"]),
                "t": t,
                "pos": np.array([float(node.get("z", 0.0)), float(node.get("y", 0.0)), float(node.get("x", 0.0))]),
                "conf": float(node.get("conf", node.get("score", 0.90))),
                "is_mitosis": float(node.get("is_mitosis", 0.0)),
            })

        for edge in data.get("edges", []):
            edges.append((int(edge["source"]), int(edge["target"])))

    except Exception as e:
        print(f"[CACHE-EXTRACTOR] Warning parsing GEFF {geff_path}: {e}")

    return {"nodes": nodes_by_t, "edges": edges}


def build_fast_loop_cache_from_predictions(
    pred_nodes_by_t: Dict[int, List[Dict[str, Any]]],
    gt_nodes_by_t: Dict[int, List[Dict[str, Any]]],
    gt_edges: List[Tuple[int, int]],
    estimated_nodes_by_t: Dict[int, int],
    output_path: Path,
) -> Path:
    """Serializes extracted predictions into Fast-Loop validation cache."""
    serializable = {
        "metadata": {
            "source": "model_inference_extraction",
            "total_raw_detections": sum(len(d) for d in pred_nodes_by_t.values()),
            "total_gt_nodes": sum(len(d) for d in gt_nodes_by_t.values()),
            "total_gt_edges": len(gt_edges),
        },
        "estimated_nodes_per_frame": {str(k): v for k, v in estimated_nodes_by_t.items()},
        "gt_edges": gt_edges,
        "gt_nodes": {
            str(t): [
                {"id": int(n["id"]), "t": int(n["t"]), "pos": [float(x) for x in n["pos"]]}
                for n in nodes
            ]
            for t, nodes in gt_nodes_by_t.items()
        },
        "raw_detections": {
            str(t): [
                {
                    "id": int(d["id"]),
                    "t": int(d["t"]),
                    "pos": [float(x) for x in d["pos"]],
                    "conf": float(d.get("conf", 0.9)),
                    "is_mitosis": float(d.get("is_mitosis", 0.0)),
                }
                for d in dets
            ]
            for t, dets in pred_nodes_by_t.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"[CACHE-EXTRACTOR] Compiled Fast-Loop cache to: {output_path}")
    return output_path
