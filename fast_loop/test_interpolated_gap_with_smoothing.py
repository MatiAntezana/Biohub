#!/usr/bin/env python3
"""
Test Trajectory Smoothing (W=4, alpha=0.85) combined with Synthetic Gap-2 Node Interpolation.
Runs remotely on authorized cluster node A1558.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.remote_executor import execute_remotely
if execute_remotely(file_path=__file__):
    sys.exit(0)

from core.graph_builder import (
    build_lineage_graph,
    apply_node_budgeting,
    apply_anisotropic_z_nms,
    link_frame_to_frame,
    reconcile_intra_frame_handovers,
    add_safe_mitotic_divisions,
    filter_orphan_tracks,
    SCALE_ZYX,
    physical_distance,
    CONF_THRESHOLD,
    NODE_BUDGET_FACTOR,
    ENABLE_Z_NMS,
    Z_NMS_XY_RADIUS_UM,
    Z_NMS_Z_DIST_UM,
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


def smooth_trajectories(
    detections: dict[int, list[dict]],
    edges: list[tuple[int, int]],
    window: int = 4,
    weight: float = 0.85,
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


def interpolate_gap2(
    filtered_dets: dict[int, list[dict]],
    edges: list[tuple[int, int]],
    fwd_links: dict[int, int],
    bwd_links: dict[int, int],
    max_gap_dist_um: float = 5.5,
    max_step_dist_um: float = 3.5,
) -> tuple[dict[int, list[dict]], list[tuple[int, int]], dict[int, int], dict[int, int], int]:
    new_dets = {t: list(cells) for t, cells in filtered_dets.items()}
    new_edges = list(edges)
    new_fwd = dict(fwd_links)
    new_bwd = dict(bwd_links)

    frames = sorted(filtered_dets.keys())
    next_id = 90000000
    interpolated_count = 0

    for i in range(len(frames) - 2):
        t1, t2, t3 = frames[i], frames[i + 1], frames[i + 2]
        if t3 != t1 + 2 or t2 != t1 + 1:
            continue

        dead_ends = [n for n in new_dets[t1] if n["id"] not in new_fwd]
        new_starts = [n for n in new_dets[t3] if n["id"] not in new_bwd]

        if not dead_ends or not new_starts:
            continue

        p_pos = np.array([de["pos"] for de in dead_ends], dtype=np.float64) * SCALE_ZYX
        c_pos = np.array([ns["pos"] for ns in new_starts], dtype=np.float64) * SCALE_ZYX

        dists = np.linalg.norm(p_pos[:, None, :] - c_pos[None, :, :], axis=-1)
        cost = np.where(dists <= max_gap_dist_um, dists, 1e6)

        r_ind, c_ind = linear_sum_assignment(cost)

        for r, c in zip(r_ind, c_ind):
            d = dists[r, c]
            if d <= max_gap_dist_um and (d / 2.0) <= max_step_dist_um:
                p_node = dead_ends[r]
                c_node = new_starts[c]
                pid = p_node["id"]
                cid = c_node["id"]

                mid_pos = (np.array(p_node["pos"], dtype=np.float64) + np.array(c_node["pos"], dtype=np.float64)) / 2.0
                mid_id = next_id
                next_id += 1

                mid_node = {
                    "id": mid_id,
                    "t": int(t2),
                    "pos": mid_pos.tolist(),
                    "conf": 0.99,
                    "is_mitosis": 0.0,
                    "synthetic": True,
                }
                new_dets[t2].append(mid_node)
                new_edges.append((pid, mid_id))
                new_edges.append((mid_id, cid))
                new_fwd[pid] = mid_id
                new_fwd[mid_id] = cid
                new_bwd[mid_id] = pid
                new_bwd[cid] = mid_id
                interpolated_count += 1

    return new_dets, new_edges, new_fwd, new_bwd, interpolated_count


def run_pipeline_with_interpolation(
    ds: dict,
    gap_dist: float = 5.5,
    step_dist: float = 3.5,
    smooth_win: int = 4,
    smooth_weight: float = 0.85,
) -> tuple[dict, list, list, int]:
    raw_dets = ds["raw_detections"]
    est_nodes = ds["estimated_nodes_per_frame"]
    edge_candidates = ds.get("edge_candidates")

    # Step 1: Cardinality budgeting
    filtered_dets = apply_node_budgeting(raw_dets, est_nodes, conf_thresh=CONF_THRESHOLD, budget_factor=NODE_BUDGET_FACTOR)

    # Step 1b: Z-NMS
    if ENABLE_Z_NMS:
        filtered_dets = apply_anisotropic_z_nms(filtered_dets, xy_radius_um=Z_NMS_XY_RADIUS_UM, z_dist_um=Z_NMS_Z_DIST_UM)

    # Step 2: Frame-to-frame LAP tracking
    edges, fwd_links, bwd_links = link_frame_to_frame(
        filtered_dets,
        max_dist_um=MAX_LINK_DISTANCE_UM,
        edge_candidates=edge_candidates,
        neural_weight=NEURAL_EDGE_WEIGHT,
        max_prob_dist_um=NEURAL_MAX_PROB_DIST_UM,
        prob_threshold=NEURAL_PROB_THRESHOLD,
    )

    # Step 2b: Handover stitching
    if ENABLE_HANDOVER_STITCHING:
        edges, fwd_links, bwd_links = reconcile_intra_frame_handovers(
            filtered_dets, edges, fwd_links, bwd_links,
            max_intra_dist_um=HANDOVER_MAX_INTRA_DIST_UM,
            max_target_dist_um=HANDOVER_MAX_TARGET_DIST_UM,
            min_history_len=HANDOVER_MIN_HISTORY_LEN,
        )

    # Step 3: Gap-2 Synthetic Node Interpolation
    interp_count = 0
    if gap_dist > 0:
        filtered_dets, edges, fwd_links, bwd_links, interp_count = interpolate_gap2(
            filtered_dets, edges, fwd_links, bwd_links,
            max_gap_dist_um=gap_dist,
            max_step_dist_um=step_dist,
        )

    # Step 4: Mitoses
    mitoses = []
    if ENABLE_SAFE_DIVISIONS:
        edges, mitoses = add_safe_mitotic_divisions(filtered_dets, edges, fwd_links, bwd_links, edge_candidates=edge_candidates)

    # Step 5: Prune short tracks
    final_dets, final_edges = filter_orphan_tracks(
        filtered_dets, edges, mitoses,
        min_track_len=OUTPUT_MIN_TRACK_LEN,
        keep_division_components=KEEP_DIVISION_COMPONENTS,
        adaptive_rescue=ADAPTIVE_SHORT_TRACK_RESCUE,
    )

    # Step 6: Trajectory smoothing
    if smooth_win > 0 and smooth_weight > 0:
        final_dets = smooth_trajectories(final_dets, final_edges, window=smooth_win, weight=smooth_weight, poly_deg=1)

    return final_dets, final_edges, mitoses, interp_count


def main():
    print("=" * 80)
    print("🔬 TESTING GAP-2 SYNTHETIC NODE INTERPOLATION + TRAJECTORY SMOOTHING")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)
    total_est = sum(ds["estimated_nodes_per_frame"].values())

    # Baseline 1: REAL-019 (no gap, no smoothing)
    dets0, edges0, mit0, _ = run_pipeline_with_interpolation(ds, gap_dist=0.0, smooth_win=0)
    res0 = evaluate_tracking(dets0, edges0, ds["gt_nodes"], ds["gt_edges"], total_est)
    print(f"Base REAL-019: Score={res0.total_score:.4f} | AdjEdge={res0.adjusted_edge_jaccard:.4f} | TP={res0.edge_tp}, FP={res0.edge_fp}, FN={res0.edge_fn}, Nodes={res0.total_predicted_nodes}")

    # Baseline 2: Smoothing Only (W=4, alpha=0.85)
    dets_sm, edges_sm, mit_sm, _ = run_pipeline_with_interpolation(ds, gap_dist=0.0, smooth_win=4, smooth_weight=0.85)
    res_sm = evaluate_tracking(dets_sm, edges_sm, ds["gt_nodes"], ds["gt_edges"], total_est)
    print(f"Smoothing W=4: Score={res_sm.total_score:.4f} | AdjEdge={res_sm.adjusted_edge_jaccard:.4f} | TP={res_sm.edge_tp}, FP={res_sm.edge_fp}, FN={res_sm.edge_fn}")

    # Sweep Gap-2 Interpolation thresholds
    gap_thresholds = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    step_thresholds = [2.5, 3.0, 3.5]

    best_score = res_sm.total_score
    best_config = None

    print("\n--- SWEEPING GAP-2 INTERPOLATION ---")
    for g_dist in gap_thresholds:
        for s_dist in step_thresholds:
            d_i, e_i, m_i, count = run_pipeline_with_interpolation(
                ds, gap_dist=g_dist, step_dist=s_dist, smooth_win=4, smooth_weight=0.85
            )
            res_i = evaluate_tracking(d_i, e_i, ds["gt_nodes"], ds["gt_edges"], total_est)
            delta = res_i.total_score - best_score
            star = "🔥 NEW BEST!" if res_i.total_score > best_score else ""
            print(
                f"Gap={g_dist:.1f}um, Step={s_dist:.1f}um | Interp={count:2d} | "
                f"Score={res_i.total_score:.4f} ({delta:+.4f}) | "
                f"AdjEdge={res_i.adjusted_edge_jaccard:.4f} | "
                f"TP={res_i.edge_tp}, FP={res_i.edge_fp}, FN={res_i.edge_fn}, Nodes={res_i.total_predicted_nodes} {star}"
            )
            if res_i.total_score > best_score:
                best_score = res_i.total_score
                best_config = (g_dist, s_dist, count)

    if best_config:
        print(f"\n🏆 CHAMPION RESULT: Score={best_score:.4f} with Gap={best_config[0]}um, Step={best_config[1]}um ({best_config[2]} nodes interpolated)")
    else:
        print(f"\nSmoothing W=4 alone remains the champion at {best_score:.4f}")


if __name__ == "__main__":
    main()
