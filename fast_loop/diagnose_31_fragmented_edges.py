#!/usr/bin/env python3
"""
Forensic diagnosis of the 31 remaining fragmented edges in REAL-020 (0.9958).
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
    build_lineage_graph,
    SCALE_ZYX,
    physical_distance,
)
from core.diagnostic_evaluator import match_nodes_bipartite
from fast_loop.mock_cache import load_dataset_from_cache


def main():
    print("=" * 80)
    print("🔬 FORENSIC INVESTIGATION: 31 FRAGMENTED EDGES (REAL-020)")
    print("=" * 80)

    cache_path = PROJECT_ROOT / "fast_loop" / "val_cache.json"
    ds = load_dataset_from_cache(cache_path)

    dets, edges, mitoses = build_lineage_graph(
        ds["raw_detections"],
        ds["estimated_nodes_per_frame"],
        edge_candidates=ds.get("edge_candidates"),
    )

    pred_edge_set = set(edges)
    pred_by_id = {c["id"]: c for cell_list in dets.values() for c in cell_list}
    gt_by_id = {c["id"]: c for cell_list in ds["gt_nodes"].values() for c in cell_list}

    # Bipartite match at each frame
    pred_to_gt = {}
    gt_to_pred = {}
    for t in sorted(ds["gt_nodes"].keys()):
        p_nodes = dets.get(t, [])
        g_nodes = ds["gt_nodes"].get(t, [])
        p2g, g2p = match_nodes_bipartite(p_nodes, g_nodes, scale=SCALE_ZYX, max_dist_um=7.0)
        pred_to_gt.update(p2g)
        gt_to_pred.update(g2p)

    gt_edges = [(int(u), int(v)) for u, v in ds["gt_edges"]]

    broken_edges = []
    lost_to_det = []

    for gu, gv in gt_edges:
        pu = gt_to_pred.get(gu)
        pv = gt_to_pred.get(gv)

        if pu is None or pv is None:
            lost_to_det.append((gu, gv, pu, pv))
            continue

        if (pu, pv) not in pred_edge_set:
            gu_node = gt_by_id[gu]
            gv_node = gt_by_id[gv]
            pu_node = pred_by_id[pu]
            pv_node = pred_by_id[pv]

            dist_gt = physical_distance(gu_node["pos"], gv_node["pos"])
            dist_pred = physical_distance(pu_node["pos"], pv_node["pos"])

            # Check if pu or pv are linked to other nodes
            pu_targets = [v for u, v in edges if u == pu]
            pv_sources = [u for u, v in edges if v == pv]

            broken_edges.append({
                "gu": gu, "gv": gv,
                "pu": pu, "pv": pv,
                "t_src": gu_node.get("t"),
                "dist_gt": dist_gt,
                "dist_pred": dist_pred,
                "pu_targets": pu_targets,
                "pv_sources": pv_sources,
            })

    print(f"Total GT edges: {len(gt_edges)}")
    print(f"Lost to detection (node missing): {len(lost_to_det)}")
    print(f"Fragmented tracking (both nodes present): {len(broken_edges)}")

    print("\n--- DISTANCE DISTRIBUTION OF FRAGMENTED EDGES ---")
    distances = [b["dist_pred"] for b in broken_edges]
    print(f"Min: {min(distances):.2f} um, Max: {max(distances):.2f} um, Mean: {np.mean(distances):.2f} um, Median: {np.median(distances):.2f} um")

    print("\n--- DETAILED BREAKDOWN OF 31 BROKEN EDGES ---")
    for i, b in enumerate(broken_edges):
        status = []
        if not b["pu_targets"]:
            status.append("pu dead-end")
        else:
            status.append(f"pu->{b['pu_targets']}")
        if not b["pv_sources"]:
            status.append("pv orphan-start")
        else:
            status.append(f"{b['pv_sources']}->pv")

        print(
            f"[{i+1:02d}] Frame {b['t_src']} | GT ({b['gu']}->{b['gv']}) | "
            f"Pred ({b['pu']}->{b['pv']}) | Dist: pred={b['dist_pred']:.2f}um, gt={b['dist_gt']:.2f}um | "
            f"{' | '.join(status)}"
        )


if __name__ == "__main__":
    main()
