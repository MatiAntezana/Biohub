#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.graph_builder import (
    apply_node_budgeting,
    apply_anisotropic_z_nms,
    reconcile_intra_frame_handovers,
    add_safe_mitotic_divisions,
    filter_orphan_tracks,
    SCALE_ZYX,
)
from core.diagnostic_evaluator import evaluate_tracking

def link_three_pass_mnn(
    filtered_detections: dict,
    edge_candidates: dict = None,
    neural_weight: float = 0.10,
    pass1_dist: float = 4.1,
    pass2_dist: float = 5.6,
    pass2_prob_thresh: float = 0.05,
    pass3_mnn_dist: float = 4.8,
) -> tuple:
    edges = []
    fwd_links = {}
    bwd_links = {}

    frames = sorted(filtered_detections.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x)

    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i + 1]
        try:
            if int(t_next) != int(t_curr) + 1:
                continue
        except (ValueError, TypeError):
            continue

        parents = filtered_detections[t_curr]
        children = filtered_detections[t_next]
        if not parents or not children:
            continue

        p_pos = np.array([p["pos"] for p in parents], dtype=np.float64) * SCALE_ZYX
        c_pos = np.array([c["pos"] for c in children], dtype=np.float64) * SCALE_ZYX
        dists = np.linalg.norm(p_pos[:, None, :] - c_pos[None, :, :], axis=-1)

        p_id_to_idx = {p["id"]: r for r, p in enumerate(parents)}
        c_id_to_idx = {c["id"]: col for col, c in enumerate(children)}

        probs = np.zeros_like(dists, dtype=np.float32)
        if edge_candidates:
            cands = edge_candidates.get(t_curr) or edge_candidates.get(str(t_curr))
            if not cands and str(t_curr).isdigit():
                cands = edge_candidates.get(int(t_curr))
            if cands:
                for c in cands:
                    sid, tid = c["src_id"], c["tgt_id"]
                    if sid in p_id_to_idx and tid in c_id_to_idx:
                        probs[p_id_to_idx[sid], c_id_to_idx[tid]] = c["prob"]

        # --- PASS 1: Tight Geometric Linking ---
        cost_p1 = dists - neural_weight * (probs * 10.0) if neural_weight > 0.0 else dists
        allowed_p1 = dists <= pass1_dist
        cost_p1 = np.where(allowed_p1, cost_p1, 1e6)
        r1, c1 = linear_sum_assignment(cost_p1)

        matched_p = set()
        matched_c = set()
        for r, c in zip(r1, c1):
            if cost_p1[r, c] < 1e5:
                pid = parents[r]["id"]
                cid = children[c]["id"]
                edges.append((pid, cid))
                fwd_links[pid] = cid
                bwd_links[cid] = pid
                matched_p.add(r)
                matched_c.add(c)

        # --- PASS 2: Residual Neural Linking ---
        unmatched_p = [r for r in range(len(parents)) if r not in matched_p]
        unmatched_c = [c for c in range(len(children)) if c not in matched_c]

        if unmatched_p and unmatched_c:
            sub_dists = dists[np.ix_(unmatched_p, unmatched_c)]
            sub_probs = probs[np.ix_(unmatched_p, unmatched_c)]
            sub_cost = sub_dists - neural_weight * (sub_probs * 10.0) if neural_weight > 0.0 else sub_dists

            allowed_p2 = (sub_dists <= pass2_dist) & (sub_probs >= pass2_prob_thresh)
            sub_cost = np.where(allowed_p2, sub_cost, 1e6)
            r2, c2 = linear_sum_assignment(sub_cost)

            for r, c in zip(r2, c2):
                if sub_cost[r, c] < 1e5:
                    pid = parents[unmatched_p[r]]["id"]
                    cid = children[unmatched_c[c]]["id"]
                    edges.append((pid, cid))
                    fwd_links[pid] = cid
                    bwd_links[cid] = pid
                    matched_p.add(unmatched_p[r])
                    matched_c.add(unmatched_c[c])

        # --- PASS 3: Mutual Nearest Neighbor Fallback ---
        if pass3_mnn_dist > 0.0:
            rem_p = [r for r in range(len(parents)) if r not in matched_p]
            rem_c = [c for c in range(len(children)) if c not in matched_c]

            if rem_p and rem_c:
                rem_dists = dists[np.ix_(rem_p, rem_c)]  # shape: (len(rem_p), len(rem_c))

                # For each remaining parent, find closest remaining child
                best_c_for_p = np.argmin(rem_dists, axis=1)
                # For each remaining child, find closest remaining parent
                best_p_for_c = np.argmin(rem_dists, axis=0)

                for r_idx, c_idx in enumerate(best_c_for_p):
                    # Check mutual agreement: child's best parent is this parent
                    if best_p_for_c[c_idx] == r_idx:
                        d = rem_dists[r_idx, c_idx]
                        if d <= pass3_mnn_dist:
                            pid = parents[rem_p[r_idx]]["id"]
                            cid = children[rem_c[c_idx]]["id"]
                            edges.append((pid, cid))
                            fwd_links[pid] = cid
                            bwd_links[cid] = pid

    return edges, fwd_links, bwd_links

def main():
    cache_file = PROJECT_ROOT / 'fast_loop' / 'val_cache.json'
    with open(cache_file, 'r', encoding='utf-8') as f:
        ds = json.load(f)

    raw_int = {int(k): v for k, v in ds['raw_detections'].items()}
    est_int = {int(k): v for k, v in ds['estimated_nodes_per_frame'].items()}
    total_est = sum(est_int.values())
    gt_nodes_int = {int(k): v for k, v in ds['gt_nodes'].items()}
    gt_edges_tuples = [(int(u), int(v)) for u, v in ds['gt_edges']]
    edge_cands = ds.get('edge_candidates', {})

    base_score = 0.9493
    print(f"Testing Three-Pass MNN Linking against baseline {base_score:.4f}...")

    filtered = apply_node_budgeting(raw_int, est_int, conf_thresh=0.960, budget_factor=1.22)
    filtered = apply_anisotropic_z_nms(filtered, xy_radius_um=3.5, z_dist_um=4.6)

    for mnn_d in [4.2, 4.4, 4.6, 4.8, 5.0, 5.2]:
        edges, fwd, bwd = link_three_pass_mnn(
            filtered,
            edge_candidates=edge_cands,
            neural_weight=0.10,
            pass1_dist=4.1,
            pass2_dist=5.6,
            pass2_prob_thresh=0.05,
            pass3_mnn_dist=mnn_d,
        )
        edges, fwd, bwd = reconcile_intra_frame_handovers(filtered, edges, fwd, bwd, max_intra_dist_um=5.5, max_target_dist_um=4.2, min_history_len=2)
        edges, mit = add_safe_mitotic_divisions(filtered, edges, fwd, bwd, edge_candidates=edge_cands)
        f_dets, f_edges = filter_orphan_tracks(filtered, edges, mit, min_track_len=6, keep_division_components=True, adaptive_rescue=True, rescue_min_len=3, rescue_max_dist_um=0.90)
        tel = evaluate_tracking(pred_nodes_by_t=f_dets, pred_edges=f_edges, gt_nodes_by_t=gt_nodes_int, gt_edges=gt_edges_tuples, estimated_total_nodes=total_est)
        sc = tel.total_score
        marker = "🔥 NEW PEAK" if sc > base_score else ("=" if sc == base_score else " ")
        print(f"{marker} mnn_d={mnn_d:.1f} -> Score: {sc:.4f} (TP={tel.edge_tp}, FP={tel.edge_fp}, FN={tel.edge_fn}, Penalty={tel.node_penalty_multiplier:.4f})")
        if sc > base_score:
            base_score = sc

if __name__ == '__main__':
    main()
