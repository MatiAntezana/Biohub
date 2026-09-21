#!/usr/bin/env python3
"""Forensic diagnosis of True Positive vs False Positive division forks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)

from core import mitosis_counterfactual as official_eval
from core.evaluation import make_graph
from core.production_adapter import as_evaluator_inputs
from core.nextgen.contracts import load_reference
from core.temporal_association_042 import config, build_candidate_graph
from geff import GeffMetadata
from ops.workflow import load_graph

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="44b6_587a1e22")
    p.add_argument("--split", default="confirmation")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--node")
    p.add_argument("--artifact-root")
    p.add_argument("--code-hash")
    args = p.parse_args()

    scale = np.array([1.625, 0.40625, 0.40625], dtype=float)
    cfg = config()
    v_cfg = cfg["variants"]["variant_1"]

    base = ROOT / "artifacts/A1688/b0970-prepare-005"
    ref_csv = base / f"cache-{args.split}/reference.csv"
    gt_dir = base / f"cache-{args.split}/raw"

    if not ref_csv.exists():
        base = Path("/home/alumno1/Documents/Biohub-0970/runs/b0970-prepare-005")
        ref_csv = base / f"cache-{args.split}/reference.csv"
        gt_dir = base / f"cache-{args.split}/raw"

    ref = load_reference(ref_csv, [args.panel])[args.panel]
    group = args.panel.split("_", 1)[0]

    cand_graph, audit = build_candidate_graph(ref, scale, None, {}, v_cfg, group=group)

    gt_path = gt_dir / f"{args.panel}.geff"
    gt = load_graph(gt_path)

    nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(cand_graph.coords)}
    edges_list = [{"source_id": int(u), "target_id": int(v)} for u, v in cand_graph.edges]
    pred = make_graph(*as_evaluator_inputs(nodes, edges_list, round_coordinates=False))

    metric = official_eval.official_metrics()
    counts = metric.evaluate(pred, gt, scale=tuple(scale))
    print(f"=== FORK DIAGNOSIS FOR {args.panel} ({group}) ===")
    print(f"Overall: Edge TP={counts.edge_tp}, FP={counts.edge_fp}, FN={counts.edge_fn}")
    print(f"Overall: Div TP={counts.division_tp}, FP={counts.division_fp}, FN={counts.division_fn}")

    children = defaultdict(list)
    parents = {}
    for u, v in cand_graph.edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    coords = cand_graph.coords
    coords_um = coords[:, 1:4] * scale


    # Inspect GT divisions directly
    div_nodes = list(gt.dividing_nodes())
    print(f"\n--- True Divisions in Ground Truth via gt.dividing_nodes(): {len(div_nodes)} ---")
    for m in div_nodes:
        parents = list(gt.predecessors(m))
        children_list = list(gt.successors(m))
        print(f"  GT Div Mother={m}: predecessors={parents}, successors={children_list}")

    divs = [(m, ch) for m, ch in children.items() if len(ch) == 2]
    print(f"Total Candidate Divisions in Candidate Graph: {len(divs)}")

    for idx, (m, (d1, d2)) in enumerate(divs):
        vm = coords_um[m]
        v1 = coords_um[d1] - vm
        v2 = coords_um[d2] - vm
        d_v1 = float(np.linalg.norm(v1))
        d_v2 = float(np.linalg.norm(v2))
        cos_3d = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        d_daughters = float(np.linalg.norm(coords_um[d1] - coords_um[d2]))

        l1, l2 = 0, 0
        c1, c2 = d1, d2
        while c1 in children and len(children[c1]) == 1:
            c1 = children[c1][0]
            l1 += 1
            if l1 > 20: break
        while c2 in children and len(children[c2]) == 1:
            c2 = children[c2][0]
            l2 += 1
            if l2 > 20: break

        mlen = 0
        cur = m
        while cur in parents:
            cur = parents[cur]
            mlen += 1
            if mlen > 20: break

        print(f"Fork #{idx + 1}: Mother={m} (t={coords[m, 0]:.0f}, mlen={mlen}) -> D1={d1} (len={l1}), D2={d2} (len={l2})")
        print(f"  cos_3d: {cos_3d:.4f}")
        print(f"  d_daughters: {d_daughters:.3f} um")
        print(f"  d1 birth: {d_v1:.3f} um, d2 birth: {d_v2:.3f} um, ratio: {min(d_v1, d_v2)/max(d_v1, d_v2):.3f}")

if __name__ == "__main__":
    main()
