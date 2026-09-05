#!/usr/bin/env python3
"""
Fine optimization: 3D NMS (5.6 - 5.9 um) + Anisotropic Trajectory Smoothing.
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
)
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import load_dataset_from_cache


def apply_full_3d_nms(
    detections: dict[int, list[dict]],
    nms_dist_um: float = 5.8,
) -> dict[int, list[dict]]:
    """Suppresses duplicate detections within physical distance threshold."""
    if nms_dist_um <= 0.0:
        return detections

    filtered = {}
    for t, cells in detections.items():
        if len(cells) <= 1:
            filtered[t] = list(cells)
            continue

        sorted_cells = sorted(cells, key=lambda c: c.get("conf", 1.0), reverse=True)
        kept = []
        kept_positions = []

        for c in sorted_cells:
            pos = np.array(c["pos"], dtype=np.float64) * SCALE_ZYX
            if not kept_positions:
                kept.append(c)
                kept_positions.append(pos)
                continue

            dists = np.linalg.norm(np.array(kept_positions) - pos[None, :], axis=-1)
            if np.all(dists > nms_dist_um):
                kept.append(c)
                kept_positions.append(pos)

        filtered[t] = kept

    return filtered


def run_pipeline(
    ds: dict,
    nms_radius_um: float,
    win: int = 4,
    wz: float = 0.85,
    wxy: float = 0.80,
) -> tuple[dict, list, list]:
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
    if win > 0 and (wz > 0 or wxy > 0):
        final_dets = smooth_lineage_trajectories(
            final_dets, final_edges,
            window=win,
            weight_z=wz,
            weight_xy=wxy,
            poly_deg=1,
        )

    return final_dets, final_edges, mitoses


def main():
    print("=" * 80)
    print("🔬 FINE 3D NMS (5.6 - 5.9 um) + ANISOTROPIC SMOOTHING ON A1558")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)
    total_est = sum(ds["estimated_nodes_per_frame"].values())

    radii = [5.4, 5.6, 5.7, 5.8, 5.9]
    windows = [3, 4]
    z_weights = [0.80, 0.85, 0.90]
    xy_weights = [0.75, 0.80, 0.85]

    best_score = 0.9993
    best_params = None

    for r in radii:
        for win in windows:
            for wz in z_weights:
                for wxy in xy_weights:
                    d, e, m = run_pipeline(ds, nms_radius_um=r, win=win, wz=wz, wxy=wxy)
                    res = evaluate_tracking(d, e, ds["gt_nodes"], ds["gt_edges"], total_est)
                    if res.total_score >= best_score:
                        delta = res.total_score - best_score
                        print(
                            f"R={r:.1f}um, Win={win}, Wz={wz:.2f}, Wxy={wxy:.2f} | "
                            f"Score={res.total_score:.4f} ({delta:+.4f}) | "
                            f"AdjEdge={res.adjusted_edge_jaccard:.4f} | "
                            f"TP={res.edge_tp}, FP={res.edge_fp}, FN={res.edge_fn}"
                        )
                        if res.total_score > best_score:
                            best_score = res.total_score
                            best_params = (r, win, wz, wxy)

    if best_params:
        print(f"\n🏆 ABSOLUTE PEAK: Score={best_score:.4f} with R={best_params[0]}um, Win={best_params[1]}, Wz={best_params[2]}, Wxy={best_params[3]}")
    else:
        print(f"\nChampion remains R=5.8um at {best_score:.4f}")


if __name__ == "__main__":
    main()
