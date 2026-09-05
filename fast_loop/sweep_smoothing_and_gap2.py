#!/usr/bin/env python3
"""
Hyperparameter Sweep for Trajectory Smoothing and Temporal Gap-2 Interpolation.
Runs remotely on authorized cluster node A1557.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.remote_executor import execute_remotely
if execute_remotely(file_path=__file__):
    sys.exit(0)

from core.graph_builder import (
    build_lineage_graph,
    SCALE_ZYX,
    physical_distance,
    OUTPUT_MIN_TRACK_LEN,
    KEEP_DIVISION_COMPONENTS,
    ADAPTIVE_SHORT_TRACK_RESCUE,
)
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import load_dataset_from_cache


def smooth_trajectories(
    detections: dict[int, list[dict]],
    edges: list[tuple[int, int]],
    window: int = 2,
    weight: float = 0.8,
    poly_deg: int = 1,
) -> dict[int, list[dict]]:
    if window <= 0 or weight <= 0:
        return detections

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for u, v in edges:
        successor.setdefault(u, []).append(v)
        predecessor.setdefault(v, []).append(u)

    nodes_dict = {c["id"]: c for cell_list in detections.values() for c in cell_list}
    orig_pos = {nid: np.array(c["pos"], dtype=np.float64) for nid, c in nodes_dict.items()}
    updated_pos: dict[int, np.ndarray] = {}
    weight_clamped = float(np.clip(weight, 0.0, 1.0))

    for nid in sorted(nodes_dict):
        nbrs = [(0, nid)]
        curr = nid
        for step in range(1, window + 1):
            prevs = predecessor.get(curr, [])
            if len(prevs) != 1:
                break
            curr = prevs[0]
            if curr not in orig_pos:
                break
            nbrs.append((-step, curr))

        curr = nid
        for step in range(1, window + 1):
            nexts = successor.get(curr, [])
            if len(nexts) != 1:
                break
            curr = nexts[0]
            if curr not in orig_pos:
                break
            nbrs.append((step, curr))

        min_points = poly_deg + 2
        if len(nbrs) < min_points:
            continue

        dts = np.array([delta for delta, _ in nbrs], dtype=np.float64)
        coords = np.stack([orig_pos[i] for _, i in nbrs])
        try:
            poly_fit = [np.polyfit(dts, coords[:, ax], poly_deg) for ax in range(3)]
            fitted = np.array([np.polyval(poly_fit[ax], 0.0) for ax in range(3)], dtype=np.float64)
            if np.isfinite(fitted).all():
                updated_pos[nid] = (1.0 - weight_clamped) * orig_pos[nid] + weight_clamped * fitted
        except Exception:
            continue

    smoothed_dets: dict[int, list[dict]] = {}
    for t, cell_list in detections.items():
        smoothed_dets[t] = []
        for c in cell_list:
            new_c = dict(c)
            if c["id"] in updated_pos:
                new_c["pos"] = list(updated_pos[c["id"]])
            smoothed_dets[t].append(new_c)

    return smoothed_dets


def main():
    print("=" * 80)
    print("🚀 RUNNING TRAJECTORY SMOOTHING & GAP-2 SWEEP ON A1557")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)
    total_est = sum(ds["estimated_nodes_per_frame"].values())

    # Build base graph
    t0 = time.time()
    base_dets, base_edges, mitoses = build_lineage_graph(
        ds["raw_detections"],
        ds["estimated_nodes_per_frame"],
        edge_candidates=ds.get("edge_candidates"),
    )
    t_base = time.time() - t0
    res_base = evaluate_tracking(base_dets, base_edges, ds["gt_nodes"], ds["gt_edges"], total_est)
    print(f"Base REAL-019 (no smoothing): Total={res_base.total_score:.4f} | AdjEdge={res_base.adjusted_edge_jaccard:.4f} | Build={t_base:.2f}s")

    # Sweep smoothing
    windows = [1, 2, 3, 4]
    weights = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    degrees = [1, 2]

    best_score = res_base.total_score
    best_config = None

    results = []
    for deg in degrees:
        for win in windows:
            for w in weights:
                sm_dets = smooth_trajectories(base_dets, base_edges, window=win, weight=w, poly_deg=deg)
                res = evaluate_tracking(sm_dets, base_edges, ds["gt_nodes"], ds["gt_edges"], total_est)
                score = res.total_score
                results.append((score, deg, win, w, res.adjusted_edge_jaccard, res.edge_tp, res.edge_fp, res.edge_fn))
                if score > best_score:
                    best_score = score
                    best_config = (deg, win, w)
                    print(f"🔥 NEW BEST: {score:.4f} (deg={deg}, win={win}, weight={w:.2f}) | TP={res.edge_tp}, FP={res.edge_fp}, FN={res.edge_fn}")

    results.sort(key=lambda x: x[0], reverse=True)
    print("\n--- TOP 10 SMOOTHING CONFIGURATIONS ---")
    for r in results[:10]:
        print(f"Score: {r[0]:.4f} | Deg={r[1]}, Win={r[2]}, Weight={r[3]:.2f} | AdjEdge={r[4]:.4f} | TP={r[5]}, FP={r[6]}, FN={r[7]}")


if __name__ == "__main__":
    main()
