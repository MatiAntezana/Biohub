#!/usr/bin/env python3
"""
Test Expanded 3D NMS (exploring radius 3.5um -> 7.5um).
Fact: In GT, NO two cells are closer than 9.48 um!
Runs remotely on authorized cluster node A1558.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.remote_executor import execute_remotely
if execute_remotely(file_path=__file__):
    sys.exit(0)

from core.graph_builder import (
    apply_node_budgeting,
    link_frame_to_frame,
    reconcile_intra_frame_handovers,
    add_safe_mitotic_divisions,
    filter_orphan_tracks,
    smooth_lineage_trajectories,
    SCALE_ZYX,
    physical_distance,
    CONF_THRESHOLD,
    NODE_BUDGET_FACTOR,
    MAX_LINK_DISTANCE_UM,
    NEURAL_EDGE_WEIGHT,
    NEURAL_MAX_PROB_DIST_UM,
    NEURAL_PROB_THRESHOLD,
    ENABLE_HANDOVER_STITCHING,
    HANDOVER_MAX_INTRA_DIST_UM,
    HANDOVER_MAX_TARGET_DIST_UM,
    HANDOVER_MIN_HISTORY_LEN,
    ENABLE_SAFE_DIVISIONS,
    OUTPUT_MIN_TRACK_LEN,
    KEEP_DIVISION_COMPONENTS,
    ADAPTIVE_SHORT_TRACK_RESCUE,
    SMOOTHING_WINDOW,
    SMOOTHING_WEIGHT_Z,
    SMOOTHING_WEIGHT_XY,
    SMOOTHING_POLY_DEG,
)
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import load_dataset_from_cache


def apply_full_3d_nms(
    detections: dict[int, list[dict]],
    nms_dist_um: float = 5.5,
) -> dict[int, list[dict]]:
    """Suppresses duplicate detections within physical distance threshold."""
    if nms_dist_um <= 0.0:
        return detections

    filtered = {}
    for t, cells in detections.items():
        if len(cells) <= 1:
            filtered[t] = list(cells)
            continue

        # Sort descending by confidence
        sorted_cells = sorted(cells, key=lambda c: c.get("conf", 1.0), reverse=True)
        kept = []
        kept_positions = []

        for c in sorted_cells:
            pos = np.array(c["pos"], dtype=np.float64) * SCALE_ZYX
            if not kept_positions:
                kept.append(c)
                kept_positions.append(pos)
                continue

            # Distance to all previously kept cells
            dists = np.linalg.norm(np.array(kept_positions) - pos[None, :], axis=-1)
            if np.all(dists > nms_dist_um):
                kept.append(c)
                kept_positions.append(pos)

        filtered[t] = kept

    return filtered


def run_pipeline(ds: dict, nms_radius_um: float) -> tuple[dict, list, list]:
    raw_dets = ds["raw_detections"]
    est_nodes = ds["estimated_nodes_per_frame"]
    edge_candidates = ds.get("edge_candidates")

    # Step 1: Budgeting
    dets = apply_node_budgeting(raw_dets, est_nodes, conf_thresh=CONF_THRESHOLD, budget_factor=NODE_BUDGET_FACTOR)

    # Step 1b: 3D NMS
    dets = apply_full_3d_nms(dets, nms_dist_um=nms_radius_um)

    # Step 2: LAP tracking
    edges, fwd, bwd = link_frame_to_frame(
        dets,
        max_dist_um=MAX_LINK_DISTANCE_UM,
        edge_candidates=edge_candidates,
        neural_weight=NEURAL_EDGE_WEIGHT,
        max_prob_dist_um=NEURAL_MAX_PROB_DIST_UM,
        prob_threshold=NEURAL_PROB_THRESHOLD,
    )

    # Step 2b: Handover
    if ENABLE_HANDOVER_STITCHING:
        edges, fwd, bwd = reconcile_intra_frame_handovers(
            dets, edges, fwd, bwd,
            max_intra_dist_um=HANDOVER_MAX_INTRA_DIST_UM,
            max_target_dist_um=HANDOVER_MAX_TARGET_DIST_UM,
            min_history_len=HANDOVER_MIN_HISTORY_LEN,
        )

    # Step 3: Mitoses
    mitoses = []
    if ENABLE_SAFE_DIVISIONS:
        edges, mitoses = add_safe_mitotic_divisions(dets, edges, fwd, bwd, edge_candidates=edge_candidates)

    # Step 4: Short track filter
    final_dets, final_edges = filter_orphan_tracks(
        dets, edges, mitoses,
        min_track_len=OUTPUT_MIN_TRACK_LEN,
        keep_division_components=KEEP_DIVISION_COMPONENTS,
        adaptive_rescue=ADAPTIVE_SHORT_TRACK_RESCUE,
    )

    # Step 5: Anisotropic trajectory smoothing
    final_dets = smooth_lineage_trajectories(
        final_dets, final_edges,
        window=SMOOTHING_WINDOW,
        weight_z=SMOOTHING_WEIGHT_Z,
        weight_xy=SMOOTHING_WEIGHT_XY,
        poly_deg=SMOOTHING_POLY_DEG,
    )

    return final_dets, final_edges, mitoses


def main():
    print("=" * 80)
    print("🔬 SWEEPING 3D NMS RADIUS ON A1558 (GT min separation = 9.48um)")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)
    total_est = sum(ds["estimated_nodes_per_frame"].values())

    # Sweep NMS radius from 0.0 to 8.0 um
    radii = [0.0, 3.5, 4.0, 4.5, 5.0, 5.2, 5.5, 5.8, 6.0, 6.5, 7.0, 7.5, 8.0]

    best_score = 0.0
    best_r = None

    for r in radii:
        d, e, m = run_pipeline(ds, nms_radius_um=r)
        res = evaluate_tracking(d, e, ds["gt_nodes"], ds["gt_edges"], total_est)
        star = "🔥 NEW BEST!" if res.total_score > best_score else ""
        print(
            f"NMS Radius={r:4.1f}um | Score={res.total_score:.4f} | "
            f"AdjEdge={res.adjusted_edge_jaccard:.4f} | "
            f"TP={res.edge_tp:3d}, FP={res.edge_fp:2d}, FN={res.edge_fn:2d}, Nodes={res.total_predicted_nodes} {star}"
        )
        if res.total_score > best_score:
            best_score = res.total_score
            best_r = r

    print(f"\n🏆 OPTIMAL 3D NMS RADIUS: {best_r} um with Score={best_score:.4f}")


if __name__ == "__main__":
    main()
