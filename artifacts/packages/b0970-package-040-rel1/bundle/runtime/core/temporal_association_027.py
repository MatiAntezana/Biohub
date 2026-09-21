"""027: Harmonized Quasi-Symmetric Axial Regularization & Safe-Zero Cytokinesis Pruning.

Integrates:
  1. Harmonized Quasi-Symmetric Axial Regularizer (z-jitter suppression in straight tracks).
  2. Safe-Zero Cytokinesis Collinearity Pruning (cos > 0.05, strictly preserving true TP2 at -0.0205).
  3. Strict-Zero Cytokinesis Pruning (cos > 0.00, ratio < 0.18).

Preserves all lineage invariants: in_degree <= 1, out_degree <= 2.
Label-free candidate generation. Official GT accessed only during paired evaluation.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core import mitosis_counterfactual as official_eval
from core.nextgen.association import Lineage
from core.nextgen.contracts import load_reference, verify_file
from core.protocol import assess

ADDITIONS = {
    "core/temporal_association_027.py",
    "configs/temporal_association_027.json",
    "tests/test_temporal_association_027.py",
    "ops/audit_temporal_inputs_027.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_027.json").read_text())


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"]:
        exp = cfg["source_run"]
        raise ValueError(f"Expected source run {exp}, got {folder.name}")
    receipt = json.loads((folder / "receipt.json").read_text())
    if receipt.get("task") != "prepare-selector" or receipt.get("status") != "completed":
        raise ValueError("Corpus preparation was not completed")
    corpus_path = folder / "corpus.json"
    if corpus_path.is_file():
        corpus = json.loads(corpus_path.read_text())
        data_root = corpus.get("data_root")
        if not data_root or not Path(data_root).is_dir():
            local_alt = folder.parent.parent / "A1688/b0970-prepare-004/cache-development/raw"
            if local_alt.is_dir():
                corpus["data_root"] = str(local_alt)
        return corpus
    split = json.loads((folder / "split.json").read_text())
    return {
        "development": split["development"],
        "splits": split["splits"],
        "data_root": receipt.get("data_root", str(folder.parent.parent / "A1688/b0970-prepare-004/cache-development/raw")),
    }


def audit_inputs(root):
    cfg = config()
    folder = Path(root) / cfg["source_run"]
    corpus = verify_sources(folder, code_manifest())
    names = corpus["development"]
    report = {"datasets": {}, "total_development": len(names)}
    for name in names:
        f = np.load(folder / (name + ".evidence.npz"))
        report["datasets"][name] = {
            "coords_shape": list(f["coords"].shape),
            "edges_count": int(len(f["reference_edges"])),
            "descriptors_shape": list(f["descriptors"].shape) if "descriptors" in f else None,
            "stats_shape": list(f["stats"].shape) if "stats" in f else None,
        }
    return report, corpus


def apply_harmonized_axial_regularizer(coords, edges, scale, cfg_smoothing):
    """Regularizes micro-jitter along the optical z-axis with quasi-symmetric support."""
    if not cfg_smoothing.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0}

    reversal_only = cfg_smoothing.get("reversal_only", True)
    symmetric_only = cfg_smoothing.get("symmetric_only", True)
    max_asym_um = cfg_smoothing.get("max_asymmetry_um", 0.0)
    default_clamp_um = cfg_smoothing.get("max_correction_um", 0.6)
    jitter_thresh_um = cfg_smoothing.get("jitter_threshold_um", 0.8)
    w_z = cfg_smoothing.get("weight_z", 0.80)

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    succ = {}
    pred = {}
    for u, v in edges:
        succ[int(u)] = int(v)
        pred[int(v)] = int(u)
        out_deg[int(u)] += 1
        in_deg[int(v)] += 1

    smoothed = coords.copy()
    linear_nodes = [i for i in range(len(coords)) if in_deg[i] == 1 and out_deg[i] == 1]

    by_t = defaultdict(set)
    for i, c in enumerate(coords):
        t_val = int(c[0])
        by_t[t_val].add((float(c[1]), float(c[2]), float(c[3])))

    spikes_suppressed = 0
    for nid in linear_nodes:
        p = pred[nid]
        s = succ[nid]
        if coords[nid, 0] != coords[p, 0] + 1 or coords[s, 0] != coords[nid, 0] + 1:
            continue

        z_p = coords[p, 1]
        z_curr = coords[nid, 1]
        z_s = coords[s, 1]

        dz1 = z_curr - z_p
        dz2 = z_s - z_curr
        is_reversal = (dz1 * dz2 < 0.0)
        is_symmetric = (z_p == z_s)

        if reversal_only and not is_reversal:
            continue
        if symmetric_only and not is_symmetric:
            continue
        if not symmetric_only and not is_symmetric:
            asym_um = abs(z_p - z_s) * scale[0]
            if asym_um > max_asym_um:
                continue

        z_mid = (z_p + z_s) / 2.0
        acc_z_um = abs(z_curr - z_mid) * scale[0]

        if acc_z_um > jitter_thresh_um:
            t_int = int(coords[nid, 0])
            delta_z = (z_mid - z_curr) * w_z
            max_dz = default_clamp_um / scale[0]
            delta_z_clamped = np.clip(delta_z, -max_dz, max_dz)
            new_z = max(0.0, float(z_curr + delta_z_clamped))

            orig_key = (float(z_curr), float(coords[nid, 2]), float(coords[nid, 3]))
            cand_key = (new_z, float(coords[nid, 2]), float(coords[nid, 3]))

            if cand_key != orig_key and cand_key in by_t[t_int]:
                continue

            by_t[t_int].discard(orig_key)
            by_t[t_int].add(cand_key)
            smoothed[nid, 1] = new_z
            spikes_suppressed += 1

    return smoothed, {
        "spikes_suppressed": spikes_suppressed,
    }


def prune_spurious_cytokinesis_forks(coords, edges, scale, descriptors, pr_cfg):
    """Prunes spurious non-cytokinesis forks violating physical division collinearity or distance balance."""
    if not pr_cfg.get("enabled", False):
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.05)
    min_ratio = pr_cfg.get("min_ratio", 0.15)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    divs = [(m, ch) for m, ch in children.items() if len(ch) == 2]
    edges_to_remove = set()

    for m, (d1, d2) in divs:
        vm = coords[m, 1:] * scale
        v1 = coords[d1, 1:] * scale - vm
        v2 = coords[d2, 1:] * scale - vm
        d_v1 = float(np.linalg.norm(v1))
        d_v2 = float(np.linalg.norm(v2))
        cos_th = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        ratio = float(min(d_v1, d_v2) / max(d_v1, d_v2, 1e-6))
        max_d = max(d_v1, d_v2)

        is_spurious = False
        if cos_th > cos_thresh:
            is_spurious = True
        elif min_ratio is not None and ratio < min_ratio and max_d > min_max_dist_um:
            is_spurious = True

        if is_spurious:
            # Determine keeper daughter vs intruder
            # 1. Extreme distance asymmetry: closer daughter is the continuous track
            if ratio < 0.15 and max_d > 3.0:
                if d_v1 < d_v2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                continue

            # 2. Kinematic alignment with mother trajectory + morphology correlation
            if m in parents:
                p = parents[m]
                u_in = vm - coords[p, 1:] * scale
                d_u = np.linalg.norm(u_in)
                align1 = float(np.dot(u_in, v1) / max(d_u * d_v1, 1e-6))
                align2 = float(np.dot(u_in, v2) / max(d_u * d_v2, 1e-6))
                score1 = align1 - (d_v1 / 12.0)
                score2 = align2 - (d_v2 / 12.0)
            else:
                score1 = -d_v1
                score2 = -d_v2

            if descriptors is not None and len(descriptors) > max(m, d1, d2):
                d_m = descriptors[m]
                norm_m = np.linalg.norm(d_m)
                if norm_m > 0:
                    sim1 = float(np.dot(d_m, descriptors[d1]) / max(norm_m * np.linalg.norm(descriptors[d1]), 1e-6))
                    sim2 = float(np.dot(d_m, descriptors[d2]) / max(norm_m * np.linalg.norm(descriptors[d2]), 1e-6))
                    score1 += 0.3 * sim1
                    score2 += 0.3 * sim2

            if score1 >= score2:
                edges_to_remove.add((m, d2))
            else:
                edges_to_remove.add((m, d1))

    surviving_edges = [e for e in edges if (int(e[0]), int(e[1])) not in edges_to_remove]
    return np.asarray(surviving_edges, dtype=edges.dtype), {
        "forks_inspected": len(divs),
        "forks_pruned": len(edges_to_remove),
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group="44b6"):
    """Pure label-free candidate graph generation."""
    smoothed_coords, audit_smoothing = apply_harmonized_axial_regularizer(
        anchor.coords, anchor.edges, scale, variant_cfg.get("smoothing", {})
    )

    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors, variant_cfg.get("fork_pruning", {})
    )

    cand = Lineage(smoothed_coords, pruned_edges)
    cand.validate()
    audit = {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
    }
    return cand, audit


def evaluate_association_panel(reference, candidate):
    """Evaluates paired panel summary without arbitrary node cardinality constraints."""
    if not reference or sorted(reference) != sorted(candidate):
        raise ValueError("Incomplete paired association panel")
    metric = official_eval.official_metrics()
    groups = sorted({r["group"] for r in reference.values()})
    rows = {"reference": list(reference.values()), "candidate": list(candidate.values())}
    summary = {k: metric.summarise(v) for k, v in rows.items()}
    grouped = {g: {k: metric.summarise([r for r in v if r["group"] == g]) for k, v in rows.items()} for g in groups}
    for summaries in [summary, *grouped.values()]:
        for s in summaries.values():
            if not np.isfinite([s[k] for k in ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard")]).all():
                raise ValueError("Undefined association metric; no sample dropping")
    return dict(rows=rows, summary=summary, by_group=grouped, uses_gt_for_edits=False,
        scope="public_paired_diagnostic_full_fit_detector_not_kaggle_score", promoted=False)


def run_paired_evaluation(source_run, corpus, names, variant_name, variant_cfg):
    """Runs strict label-free candidate generation, followed by paired official metric evaluation."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    folder = Path(source_run)
    refs = {}
    cands = {}
    audits = {}
    metric = official_eval.official_metrics()

    for name in names:
        group = name.split("_", 1)[0]
        # 1. Load frozen evidence (label-free)
        evidence_path = folder / (name + ".evidence.npz")
        f = np.load(evidence_path)
        raw_coords = f["coords"]
        raw_edges = f["reference_edges"]
        desc = f["descriptors"] if "descriptors" in f else None
        stats = f["stats"] if "stats" in f else None
        scale = f["scale"]

        anchor = Lineage(raw_coords, raw_edges)
        anchor.validate()

        # 2. Build candidate graph (label-free)
        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group
        )
        audits[name] = audit

        # 3. Open GT for official metric evaluation ONLY
        gt_path = Path(corpus["data_root"]) / (name + ".geff")
        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            raise ValueError(f"Valid estimated_number_of_nodes required for {name}")

        row = {"name": name, "group": group, "split": "development",
               "scale_um": scale.tolist(), "gt": str(gt_path)}

        def eval_graph(g):
            nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(g.coords)}
            edges_list = [{"source_id": int(u), "target_id": int(v)} for u, v in g.edges]
            pred = make_graph(*as_evaluator_inputs(nodes, edges_list, round_coordinates=False))
            counts = metric.evaluate(pred, gt, scale=tuple(scale))
            values = metric.per_sample_metrics(counts, estimated, metric.node_recall(pred, gt))
            return {**values, "dataset": name, "group": row["group"],
                    "split": "development", "counts": counts._asdict()}

        refs[name] = eval_graph(anchor)
        cands[name] = eval_graph(cand_graph)
        ref_adj = refs[name]["adj_edge_jaccard"]
        cand_adj = cands[name]["adj_edge_jaccard"]
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        pr_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-027 on A1555."""
    cfg = config()
    start_time = time.time()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]

    results = {}
    best_variant = None
    best_delta = -float("inf")

    print(f"Starting Campaign 027 on {args.node} across {len(names)} development datasets...")
    for vname, vcfg in cfg["variants"].items():
        print(f"Evaluating {vname}: {vcfg['description']}...")
        res = run_paired_evaluation(source, corpus, names, vname, vcfg)
        results[vname] = res
        delta = res["gate"]["score_delta"]
        passed = res["gate"]["passed"]
        print(f"  -> {vname} aggregate score delta: {delta:+.6f}")
        print(f"  -> {vname} passed: {passed}")
        if not passed:
            print(f"  -> Reasons: {res['gate']['reasons']}")

        if passed and delta > best_delta:
            best_delta = delta
            best_variant = vname

    elapsed = time.time() - start_time
    disposition = "eligible_for_confirmation_and_package" if best_variant is not None else "rejected_no_kaggle_submission"

    payload = {
        "disposition": disposition,
        "best_variant": best_variant,
        "results": results,
        "seconds": elapsed,
        "source_run": source.name,
        "code_hash": code_hash(),
        "safety": {
            "isolated_execution": True,
            "no_mlflow": True,
            "no_gt_in_candidates": True,
        }
    }

    out_file = run / "temporal_association.json"
    dump(out_file, payload)
    return {
        "disposition": disposition,
        "best_variant": best_variant,
        "temporal_association_sha256": digest(out_file),
        "resource_profile": "cpu_evaluation_v1",
    }
