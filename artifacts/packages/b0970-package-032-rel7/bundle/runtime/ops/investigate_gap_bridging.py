#!/usr/bin/env python3
"""Investigate track fragmentation and 1-frame dropout gaps across the 8 development datasets.
Runs on A1555 with access to train-55 environment, raw .zarr images, and ground-truth .geff files.
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
import zarr

# biohub imports
from core.evaluation import make_graph
from core.nextgen.association import Lineage
from core.production_adapter import as_evaluator_inputs
from core.temporal_association_032 import build_candidate_graph, config
from ops.workflow import load_graph
from tracking_cellmot import metrics as official_metrics
from geff import GeffMetadata

DEV_DATASETS = [
    "44b6_d5e7d891",
    "44b6_c50204e0",
    "44b6_e57ff5c6",
    "44b6_ddf577ad",
    "6bba_48816121",
    "6bba_09961292",
    "6bba_5a4d9360",
    "6bba_07e24132",
]

CORPUS_DIR = Path("/home/alumno1/Documents/Biohub-0970/runs/b0970-selector-corpus-011")
TRAIN_DATA_DIR = Path("/home/alumno1/Documents/Biohub/data/train")


def run_investigation():
    print("=" * 80)
    print("TRACK GAP BRIDGING INVESTIGATION ACROSS 8 DEVELOPMENT DATASETS")
    print("=" * 80)

    cfg = config()
    v2_cfg = cfg["variants"]["variant_2"]

    dataset_reports = {}
    total_gt_nodes_all = 0
    total_gt_edges_all = 0
    total_pred_nodes_all = 0
    total_pred_edges_all = 0

    # Aggregate counters
    agg_gap_candidates = {
        "pairs_1_0um": 0,
        "pairs_1_5um": 0,
        "pairs_2_0um": 0,
        "pairs_3_0um": 0,
        "pairs_5_0um": 0,
        "greedy_1_0um": 0,
        "greedy_1_5um": 0,
        "greedy_2_0um": 0,
        "greedy_3_0um": 0,
    }

    agg_gt_matches = {
        "tp_1_0um": 0,
        "fp_1_0um": 0,
        "tp_1_5um": 0,
        "fp_1_5um": 0,
        "tp_2_0um": 0,
        "fp_2_0um": 0,
        "tp_3_0um": 0,
        "fp_3_0um": 0,
    }

    agg_dropout_analysis = {
        "gt_2step_paths": 0,
        "gt_1frame_dropouts": 0,
        "gt_dropouts_both_endpoints_detected": 0,
        "gt_dropouts_both_endpoints_are_track_ends": 0,
        "gt_dropouts_dist_under_1_0um": 0,
        "gt_dropouts_dist_under_1_5um": 0,
        "gt_dropouts_dist_under_2_0um": 0,
        "gt_dropouts_dist_under_3_0um": 0,
        "gt_dropouts_dist_over_3_0um": 0,
    }

    for name in DEV_DATASETS:
        group = name.split("_")[0]
        print(f"\n--- Processing {name} ({group}) ---")

        # 1. Load candidate graph evidence & build Campaign 032 candidate graph
        evidence_path = CORPUS_DIR / f"{name}.evidence.npz"
        ev = np.load(evidence_path)
        raw_coords = ev["coords"]
        raw_edges = ev["reference_edges"]
        scale = ev["scale"]
        desc = ev["descriptors"] if "descriptors" in ev else None
        stats = ev["stats"] if "stats" in ev else None

        anchor = Lineage(raw_coords, raw_edges)
        cand, audit = build_candidate_graph(anchor, scale, desc, stats, v2_cfg, group=group)

        coords = cand.coords
        edges = cand.edges
        N = len(coords)
        E = len(edges)
        print(f"  Candidate nodes: {N}, edges: {E} (pruned: {audit['fork_pruning']['forks_pruned']})")

        # 2. Load Ground Truth
        gt_path = TRAIN_DATA_DIR / f"{name}.geff"
        gt = load_graph(gt_path)
        gt_num_nodes = gt.num_nodes()
        gt_num_edges = gt.num_edges()
        total_gt_nodes_all += gt_num_nodes
        total_gt_edges_all += gt_num_edges
        total_pred_nodes_all += N
        total_pred_edges_all += E
        print(f"  Ground Truth nodes: {gt_num_nodes}, edges: {gt_num_edges}")

        # 3. Match candidate nodes to GT nodes using DistanceMatching
        nodes_dict = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(coords)}
        links_list = [{"source_id": int(u), "target_id": int(v)} for u, v in edges]
        pred_graph = make_graph(*as_evaluator_inputs(nodes_dict, links_list, round_coordinates=False))

        matching = DistanceMatching(max_distance=7.0, scale=tuple(scale))
        pred_graph.match(gt, matching=matching)

        matched_attrs = pred_graph.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        ).to_dicts()

        # Map: pred_node_id -> gt_node_id (if matched)
        pred_to_gt = {}
        gt_to_pred = {}
        for r in matched_attrs:
            pid = int(r[td.DEFAULT_ATTR_KEYS.NODE_ID])
            gid = r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
            if gid is not None and gid != -1:
                gid = int(gid)
                pred_to_gt[pid] = gid
                gt_to_pred[gid] = pid

        # 4. Build GT topology
        gt_children = defaultdict(list)
        gt_parents = {}
        gt_edge_set = set()
        for u, v in gt.edge_list():
            gt_children[int(u)].append(int(v))
            gt_parents[int(v)] = int(u)
            gt_edge_set.add((int(u), int(v)))

        gt_nodes_attrs = {int(r["node_id"]): r for r in gt.node_attrs().to_dicts()}

        # 5. Track analysis on candidate graph
        cand_children = defaultdict(list)
        cand_parents = {}
        for u, v in edges:
            cand_children[int(u)].append(int(v))
            cand_parents[int(v)] = int(u)

        times = coords[:, 0].astype(int)
        max_t = int(times.max())

        ends_by_t = defaultdict(list)
        starts_by_t = defaultdict(list)
        for i in range(N):
            t = times[i]
            if len(cand_children[i]) == 0:
                ends_by_t[t].append(i)
            if i not in cand_parents:
                starts_by_t[t].append(i)

        total_ends = sum(len(v) for v in ends_by_t.values())
        total_starts = sum(len(v) for v in starts_by_t.values())

        # 6. Gap Candidate Pairs (end at t, start at t+2)
        candidate_pairs = []
        for t in range(max_t - 1):
            ends = ends_by_t[t]
            starts = starts_by_t[t + 2]
            if not ends or not starts:
                continue
            pos_ends = coords[ends, 1:] * scale
            pos_starts = coords[starts, 1:] * scale
            diff = pos_ends[:, None, :] - pos_starts[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)  # shape: (len(ends), len(starts))

            for ei, u in enumerate(ends):
                for si, v in enumerate(starts):
                    d = float(dist[ei, si])
                    if d <= 5.0:  # record all within 5 um
                        candidate_pairs.append((u, v, t, d))

        # Sort candidate pairs by displacement Delta r
        candidate_pairs.sort(key=lambda x: x[3])

        # Measure counts in thresholds
        p_1_0 = [p for p in candidate_pairs if p[3] < 1.0]
        p_1_5 = [p for p in candidate_pairs if p[3] < 1.5]
        p_2_0 = [p for p in candidate_pairs if p[3] < 2.0]
        p_3_0 = [p for p in candidate_pairs if p[3] < 3.0]
        p_5_0 = candidate_pairs

        # Greedy non-conflicting bipartite matching for each threshold
        def greedy_match(pairs, max_d):
            used_ends = set()
            used_starts = set()
            matched = []
            for u, v, t, d in pairs:
                if d <= max_d and u not in used_ends and v not in used_starts:
                    used_ends.add(u)
                    used_starts.add(v)
                    matched.append((u, v, t, d))
            return matched

        gm_1_0 = greedy_match(candidate_pairs, 1.0)
        gm_1_5 = greedy_match(candidate_pairs, 1.5)
        gm_2_0 = greedy_match(candidate_pairs, 2.0)
        gm_3_0 = greedy_match(candidate_pairs, 3.0)

        agg_gap_candidates["pairs_1_0um"] += len(p_1_0)
        agg_gap_candidates["pairs_1_5um"] += len(p_1_5)
        agg_gap_candidates["pairs_2_0um"] += len(p_2_0)
        agg_gap_candidates["pairs_3_0um"] += len(p_3_0)
        agg_gap_candidates["pairs_5_0um"] += len(p_5_0)

        agg_gap_candidates["greedy_1_0um"] += len(gm_1_0)
        agg_gap_candidates["greedy_1_5um"] += len(gm_1_5)
        agg_gap_candidates["greedy_2_0um"] += len(gm_2_0)
        agg_gap_candidates["greedy_3_0um"] += len(gm_3_0)

        # 7. Check Ground Truth Match for candidate pairs
        # Does (u, v) belong to the SAME ground-truth cell track?
        # In GT, is there a 2-step path g(u) -> g_mid -> g(v)?
        def check_gt_same_track(u, v):
            g_u = pred_to_gt.get(u)
            g_v = pred_to_gt.get(v)
            if g_u is None or g_v is None:
                return False, "unmatched_endpoint", None
            # Check if g_v is reachable from g_u in 2 steps
            for mid in gt_children.get(g_u, []):
                if g_v in gt_children.get(mid, []):
                    return True, "same_gt_track_2step", mid
            # Check if they are connected through division or longer path
            return False, "different_gt_cells", None

        def evaluate_pair_set(pairs):
            tp = 0
            fp = 0
            details = []
            for u, v, t, d in pairs:
                is_same, reason, mid = check_gt_same_track(u, v)
                if is_same:
                    tp += 1
                else:
                    fp += 1
                details.append({
                    "u": int(u), "v": int(v), "t": int(t), "dist_um": float(d),
                    "is_same_gt_track": is_same, "reason": reason, "gt_mid": mid,
                    "gt_u": pred_to_gt.get(u), "gt_v": pred_to_gt.get(v),
                })
            return tp, fp, details

        tp_1_0, fp_1_0, det_1_0 = evaluate_pair_set(gm_1_0)
        tp_1_5, fp_1_5, det_1_5 = evaluate_pair_set(gm_1_5)
        tp_2_0, fp_2_0, det_2_0 = evaluate_pair_set(gm_2_0)
        tp_3_0, fp_3_0, det_3_0 = evaluate_pair_set(gm_3_0)

        agg_gt_matches["tp_1_0um"] += tp_1_0
        agg_gt_matches["fp_1_0um"] += fp_1_0
        agg_gt_matches["tp_1_5um"] += tp_1_5
        agg_gt_matches["fp_1_5um"] += fp_1_5
        agg_gt_matches["tp_2_0um"] += tp_2_0
        agg_gt_matches["fp_2_0um"] += fp_2_0
        agg_gt_matches["tp_3_0um"] += tp_3_0
        agg_gt_matches["fp_3_0um"] += fp_3_0

        # 8. Comprehensive Ground Truth Dropout Audit:
        # Trace ALL 2-step paths in GT: g1 -> g_mid -> g2
        # Check how many g_mid were NOT detected (1-frame dropouts)
        gt_dropouts_dataset = []
        for g1 in gt_nodes_attrs:
            for g_mid in gt_children.get(g1, []):
                for g2 in gt_children.get(g_mid, []):
                    agg_dropout_analysis["gt_2step_paths"] += 1
                    # Check if g_mid was detected in prediction
                    mid_detected = g_mid in gt_to_pred
                    if not mid_detected:
                        agg_dropout_analysis["gt_1frame_dropouts"] += 1
                        g1_pred = gt_to_pred.get(g1)
                        g2_pred = gt_to_pred.get(g2)

                        g1_pos = np.array([gt_nodes_attrs[g1]["z"], gt_nodes_attrs[g1]["y"], gt_nodes_attrs[g1]["x"]]) * scale
                        g2_pos = np.array([gt_nodes_attrs[g2]["z"], gt_nodes_attrs[g2]["y"], gt_nodes_attrs[g2]["x"]]) * scale
                        disp_gt = float(np.linalg.norm(g2_pos - g1_pos))

                        t1 = gt_nodes_attrs[g1]["t"]
                        info = {
                            "gt_g1": g1, "gt_mid": g_mid, "gt_g2": g2, "t": t1,
                            "disp_gt_um": disp_gt,
                            "g1_detected": g1_pred is not None,
                            "g2_detected": g2_pred is not None,
                            "g1_pred": g1_pred,
                            "g2_pred": g2_pred,
                            "g1_is_track_end": (g1_pred is not None and len(cand_children[g1_pred]) == 0),
                            "g2_is_track_start": (g2_pred is not None and g2_pred not in cand_parents),
                        }

                        if g1_pred is not None and g2_pred is not None:
                            agg_dropout_analysis["gt_dropouts_both_endpoints_detected"] += 1
                            if info["g1_is_track_end"] and info["g2_is_track_start"]:
                                agg_dropout_analysis["gt_dropouts_both_endpoints_are_track_ends"] += 1

                            pred_disp = float(np.linalg.norm((coords[g2_pred, 1:] - coords[g1_pred, 1:]) * scale))
                            info["pred_disp_um"] = pred_disp
                            if pred_disp < 1.0:
                                agg_dropout_analysis["gt_dropouts_dist_under_1_0um"] += 1
                            elif pred_disp < 1.5:
                                agg_dropout_analysis["gt_dropouts_dist_under_1_5um"] += 1
                            elif pred_disp < 2.0:
                                agg_dropout_analysis["gt_dropouts_dist_under_2_0um"] += 1
                            elif pred_disp < 3.0:
                                agg_dropout_analysis["gt_dropouts_dist_under_3_0um"] += 1
                            else:
                                agg_dropout_analysis["gt_dropouts_dist_over_3_0um"] += 1

                        gt_dropouts_dataset.append(info)

        # 9. Evaluate Image Evidence for Dropouts
        # Load .zarr raw image
        z_path = TRAIN_DATA_DIR / f"{name}.zarr"
        image_evidence_samples = []
        if z_path.exists():
            z_arr = zarr.open(z_path, mode="r")["0"]  # shape (T, Z, Y, X)
            # Sample first 5 dropouts where g1 and g2 were detected
            eval_dropouts = [d for d in gt_dropouts_dataset if d["g1_detected"] and d["g2_detected"]][:5]
            for drop in eval_dropouts:
                t_mid = drop["t"] + 1
                g_mid = drop["gt_mid"]
                # GT position of missed cell in voxels:
                gt_z = gt_nodes_attrs[g_mid]["z"]
                gt_y = gt_nodes_attrs[g_mid]["y"]
                gt_x = gt_nodes_attrs[g_mid]["x"]

                # Interpolated position from prediction endpoints
                p1 = coords[drop["g1_pred"], 1:]
                p2 = coords[drop["g2_pred"], 1:]
                interp_vox = 0.5 * (p1 + p2)
                iz, iy, ix = int(round(interp_vox[0])), int(round(interp_vox[1])), int(round(interp_vox[2]))

                # Extract patch at t_mid around interp_vox: 7x15x15
                z_min, z_max = max(0, iz - 3), min(z_arr.shape[1], iz + 4)
                y_min, y_max = max(0, iy - 7), min(z_arr.shape[2], iy + 8)
                x_min, x_max = max(0, ix - 7), min(z_arr.shape[3], ix + 8)

                patch = np.array(z_arr[t_mid, z_min:z_max, y_min:y_max, x_min:x_max], dtype=float)
                patch_max = float(patch.max()) if patch.size > 0 else 0.0
                patch_mean = float(patch.mean()) if patch.size > 0 else 0.0
                patch_std = float(patch.std()) if patch.size > 0 else 1.0

                # Interp point intensity
                interp_val = float(z_arr[t_mid, min(max(0, iz), z_arr.shape[1]-1),
                                         min(max(0, iy), z_arr.shape[2]-1),
                                         min(max(0, ix), z_arr.shape[3]-1)])
                # GT point intensity
                gt_val = float(z_arr[t_mid, min(max(0, int(round(gt_z))), z_arr.shape[1]-1),
                                     min(max(0, int(round(gt_y))), z_arr.shape[2]-1),
                                     min(max(0, int(round(gt_x))), z_arr.shape[3]-1)])

                snr = (patch_max - patch_mean) / max(patch_std, 1e-3)
                image_evidence_samples.append({
                    "t_mid": t_mid,
                    "gt_mid": g_mid,
                    "gt_coords_vox": [float(gt_z), float(gt_y), float(gt_x)],
                    "interp_coords_vox": [float(iz), float(iy), float(ix)],
                    "interp_val": interp_val,
                    "gt_val": gt_val,
                    "patch_max": patch_max,
                    "patch_mean": patch_mean,
                    "snr": snr,
                    "has_intensity_peak": bool(snr > 3.0),
                })

        dataset_reports[name] = {
            "group": group,
            "pred_nodes": N,
            "pred_edges": E,
            "gt_nodes": gt_num_nodes,
            "gt_edges": gt_num_edges,
            "cand_track_ends": total_ends,
            "cand_track_starts": total_starts,
            "gap_pairs_dist": {
                "<1.0um": len(p_1_0),
                "<1.5um": len(p_1_5),
                "<2.0um": len(p_2_0),
                "<3.0um": len(p_3_0),
                "<5.0um": len(p_5_0),
            },
            "greedy_matches": {
                "1.0um": {"matched": len(gm_1_0), "tp": tp_1_0, "fp": fp_1_0},
                "1.5um": {"matched": len(gm_1_5), "tp": tp_1_5, "fp": fp_1_5},
                "2.0um": {"matched": len(gm_2_0), "tp": tp_2_0, "fp": fp_2_0},
                "3.0um": {"matched": len(gm_3_0), "tp": tp_3_0, "fp": fp_3_0},
            },
            "gt_dropout_count": len(gt_dropouts_dataset),
            "image_evidence_samples": image_evidence_samples,
        }

        print(f"  Gap pairs: <1.0um: {len(p_1_0)}, <1.5um: {len(p_1_5)}, <2.0um: {len(p_2_0)}, <3.0um: {len(p_3_0)}")
        print(f"  Greedy matches (<=1.5um): {len(gm_1_5)} (TP: {tp_1_5}, FP: {fp_1_5})")
        print(f"  GT 1-frame dropouts in this dataset: {len(gt_dropouts_dataset)}")

    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS ACROSS ALL 8 DATASETS")
    print("=" * 80)
    print("1. Gap Candidates (Candidate track end at t, start at t+2):")
    print(json.dumps(agg_gap_candidates, indent=2))
    print("\n2. Ground Truth Match of Gap Candidates:")
    print(json.dumps(agg_gt_matches, indent=2))
    print("\n3. Ground Truth 1-Frame Dropout Audit:")
    print(json.dumps(agg_dropout_analysis, indent=2))

    # Save full results to JSON
    out_path = Path("/home/alumno1/Documents/Biohub-0970/runs/investigation_gap_bridging.json")
    full_output = {
        "dataset_reports": dataset_reports,
        "agg_gap_candidates": agg_gap_candidates,
        "agg_gt_matches": agg_gt_matches,
        "agg_dropout_analysis": agg_dropout_analysis,
    }
    out_path.write_text(json.dumps(full_output, indent=2))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    run_investigation()
