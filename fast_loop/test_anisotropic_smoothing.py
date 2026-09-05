#!/usr/bin/env python3
"""
Test Anisotropic Trajectory Smoothing (Z vs XY noise differential).
Runs remotely on authorized cluster node A1558.
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
)
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import load_dataset_from_cache


def smooth_trajectories_anisotropic(
    detections: dict[int, list[dict]],
    edges: list[tuple[int, int]],
    window: int = 4,
    weight_z: float = 0.90,
    weight_xy: float = 0.80,
    poly_deg: int = 1,
) -> dict[int, list[dict]]:
    if window <= 0 or (weight_z <= 0 and weight_xy <= 0):
        return detections

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for u, v in edges:
        successor.setdefault(u, []).append(v)
        predecessor.setdefault(v, []).append(u)

    nodes_dict = {c["id"]: c for cell_list in detections.values() for c in cell_list}
    orig_pos = {nid: np.array(c["pos"], dtype=np.float64) for nid, c in nodes_dict.items()}
    updated_pos: dict[int, np.ndarray] = {}
    
    weights = np.array([weight_z, weight_xy, weight_xy], dtype=np.float64)
    weights = np.clip(weights, 0.0, 1.0)

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
                updated_pos[nid] = (1.0 - weights) * orig_pos[nid] + weights * fitted
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
    print("🔬 TESTING ANISOTROPIC TRAJECTORY SMOOTHING (Z vs XY) ON A1558")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)
    total_est = sum(ds["estimated_nodes_per_frame"].values())

    # Build base graph
    base_dets, base_edges, mitoses = build_lineage_graph(
        ds["raw_detections"],
        ds["estimated_nodes_per_frame"],
        edge_candidates=ds.get("edge_candidates"),
    )
    res_base = evaluate_tracking(base_dets, base_edges, ds["gt_nodes"], ds["gt_edges"], total_est)
    print(f"Base REAL-019: Score={res_base.total_score:.4f} | TP={res_base.edge_tp}, FP={res_base.edge_fp}, FN={res_base.edge_fn}")

    # Isotropic champion: W=4, weight=0.85 -> 0.9947
    dets_iso = smooth_trajectories_anisotropic(base_dets, base_edges, window=4, weight_z=0.85, weight_xy=0.85)
    res_iso = evaluate_tracking(dets_iso, base_edges, ds["gt_nodes"], ds["gt_edges"], total_est)
    print(f"Isotropic W=4, weight=0.85: Score={res_iso.total_score:.4f} | TP={res_iso.edge_tp}, FP={res_iso.edge_fp}, FN={res_iso.edge_fn}")

    # Sweep anisotropic combinations
    z_weights = [0.80, 0.85, 0.90, 0.95, 1.00]
    xy_weights = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    windows = [3, 4, 5]

    best_score = res_iso.total_score
    best_cfg = (4, 0.85, 0.85)

    print("\n--- ANISOTROPIC SWEEP ---")
    for win in windows:
        for wz in z_weights:
            for wxy in xy_weights:
                sm_dets = smooth_trajectories_anisotropic(base_dets, base_edges, window=win, weight_z=wz, weight_xy=wxy)
                res = evaluate_tracking(sm_dets, base_edges, ds["gt_nodes"], ds["gt_edges"], total_est)
                if res.total_score > best_score:
                    best_score = res.total_score
                    best_cfg = (win, wz, wxy)
                    print(f"🔥 NEW BEST: {best_score:.4f} (Win={win}, Wz={wz:.2f}, Wxy={wxy:.2f}) | TP={res.edge_tp}, FP={res.edge_fp}, FN={res.edge_fn}")

    print(f"\n🏆 CHAMPION: Score={best_score:.4f} (Win={best_cfg[0]}, Wz={best_cfg[1]:.2f}, Wxy={best_cfg[2]:.2f})")


if __name__ == "__main__":
    main()
