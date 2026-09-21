"""024: Biomechanically Constrained Temporal Association and Cytokinesis Symmetry.

Combines:
  1. Pure Symmetric Axial Regularizer (z-jitter suppression in isolated straight trajectories).
  2. Spurious Mitosis Pruning (anti-cytokinesis divergence and extreme distance asymmetry).
  3. High-Confidence Antiparallel Cytokinesis Reconnection (recovering missed cytokinesis daughters).

Label-Free Candidate Generation:
  Inference uses only predicted coordinates, graph topology, physical scale, and 3D patch descriptors.
  Degree invariants preserved: in_degree <= 1, out_degree <= 2.
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
    "core/temporal_association_024.py",
    "configs/temporal_association_024.json",
    "tests/test_temporal_association_024.py",
    "ops/audit_temporal_inputs_024.py",
}



def config():
    return json.loads((ROOT / "configs/temporal_association_024.json").read_text())


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
            "descriptors_shape": list(f["descriptors"].shape),
            "stats_shape": list(f["stats"].shape),
        }
    return report, corpus


def apply_symmetric_axial_regularizer(coords, edges, scale, cfg_smoothing):
    """Regularizes micro-jitter along the optical z-axis in verified linear tracks."""
    if not cfg_smoothing.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0}

    reversal_only = cfg_smoothing.get("reversal_only", True)
    symmetric_only = cfg_smoothing.get("symmetric_only", True)
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
    """Prunes spurious non-cytokinesis forks violating physical division geometry or distance balance."""
    if not pr_cfg.get("enabled", False):
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.20)
    min_ratio = pr_cfg.get("min_ratio", 0.15)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)
    persistence_guard = pr_cfg.get("persistence_guard", False)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    def forward_len(start):
        c = start
        l = 1
        while children[c]:
            c = children[c][0]
            l += 1
        return l

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
        reason = None
        if cos_th > cos_thresh:
            is_spurious = True
            reason = "collinear_divergence"
        elif min_ratio is not None and ratio < min_ratio and max_d > min_max_dist_um:
            is_spurious = True
            reason = "extreme_asymmetry"

        if is_spurious:
            # Determine keeper vs intruder
            # 1. If persistence guard is enabled and one track clearly outlasts the other
            fl1 = forward_len(d1)
            fl2 = forward_len(d2)
            if persistence_guard and abs(fl1 - fl2) >= 5:
                if fl1 > fl2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                continue

            # 2. In extreme distance asymmetry where ratio < 0.15 and max/min > 4.0,
            # the closer daughter is unequivocally the continuing track
            if ratio < 0.15 and max_d > 3.0:
                if d_v1 < d_v2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                continue

            # 3. Kinematic alignment with mother trajectory + morphology correlation
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


def recover_antiparallel_cytokinesis_forks(coords, edges, scale, descriptors, cyto_cfg):
    """Recovers missed biological cytokinesis divisions under strict antiparallel and persistence constraints."""
    if not cyto_cfg.get("enabled", False):
        return edges, {"candidates_evaluated": 0, "forks_recovered": 0}

    cos_max = cyto_cfg.get("cos_max", -0.85)
    min_ratio = cyto_cfg.get("min_ratio", 0.35)
    max_dist_um = cyto_cfg.get("max_dist_um", 8.0)
    max_sister_dist_um = cyto_cfg.get("max_sister_dist_um", 12.0)
    min_backward_len = cyto_cfg.get("min_backward_len", 10)
    min_forward_len = cyto_cfg.get("min_forward_len", 10)
    min_desc_corr = cyto_cfg.get("min_desc_corr", 0.40)

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for u, v in edges:
        out_edges[int(u)].append(int(v))
        in_edges[int(v)].append(int(u))

    def forward_len(start):
        c = start
        l = 1
        while out_edges[c]:
            c = out_edges[c][0]
            l += 1
        return l

    def backward_len(start):
        c = start
        l = 1
        while in_edges[c]:
            c = in_edges[c][0]
            l += 1
        return l

    single_mothers = [u for u, children in out_edges.items() if len(children) == 1 and backward_len(u) >= min_backward_len]

    t_coords = coords[:, 0].astype(int)
    max_t = int(t_coords.max())

    candidate_pairs = []
    for t in range(max_t):
        m_t = [u for u in single_mothers if t_coords[u] == t]
        if not m_t:
            continue
        orphans_t1 = [v for v in np.flatnonzero(t_coords == t + 1) if not in_edges[v] and forward_len(v) >= min_forward_len]
        if not orphans_t1:
            continue

        orphans_pts = coords[orphans_t1, 1:] * scale
        tree = cKDTree(orphans_pts)

        for m in m_t:
            d1 = out_edges[m][0]
            v1 = (coords[d1, 1:] - coords[m, 1:]) * scale
            norm1 = np.linalg.norm(v1)
            if norm1 == 0 or norm1 > max_dist_um:
                continue

            m_pt = coords[m, 1:] * scale
            hits = tree.query_ball_point(m_pt, max_dist_um)
            for h_idx in hits:
                d2 = orphans_t1[h_idx]
                v2 = (coords[d2, 1:] - coords[m, 1:]) * scale
                norm2 = np.linalg.norm(v2)
                if norm2 == 0 or norm2 > max_dist_um:
                    continue
                cos = float(np.dot(v1, v2) / (norm1 * norm2))
                ratio = float(min(norm1, norm2) / max(norm1, norm2))
                sister_dist = float(np.linalg.norm(v1 - v2))

                if descriptors is not None and len(descriptors) > max(d1, d2):
                    p1 = descriptors[d1] / (np.linalg.norm(descriptors[d1]) or 1)
                    p2 = descriptors[d2] / (np.linalg.norm(descriptors[d2]) or 1)
                    corr = float(np.dot(p1, p2))
                else:
                    corr = 1.0

                if (cos <= cos_max and ratio >= min_ratio and sister_dist <= max_sister_dist_um
                        and corr >= min_desc_corr):
                    candidate_pairs.append({
                        "mother": m, "d1": d1, "d2": d2,
                        "cos": cos, "ratio": ratio, "sister_dist": sister_dist,
                        "corr": corr,
                    })

    # Collision resolution: each orphan d2 can only be claimed by at most one mother (lowest cos)
    # Each mother m can only claim at most one orphan (lowest cos)
    candidate_pairs.sort(key=lambda x: x["cos"])
    claimed_mothers = set()
    claimed_orphans = set()
    edges_to_add = []

    for c in candidate_pairs:
        m = c["mother"]
        d2 = c["d2"]
        if m in claimed_mothers or d2 in claimed_orphans:
            continue
        claimed_mothers.add(m)
        claimed_orphans.add(d2)
        edges_to_add.append((m, d2))

    new_edges = list(edges) + edges_to_add
    return np.asarray(new_edges, dtype=edges.dtype), {
        "candidates_evaluated": len(candidate_pairs),
        "forks_recovered": len(edges_to_add),
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group="44b6"):
    """Pure label-free candidate graph generation."""
    smoothed_coords, audit_smoothing = apply_symmetric_axial_regularizer(
        anchor.coords, anchor.edges, scale, variant_cfg.get("smoothing", {})
    )

    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors, variant_cfg.get("fork_pruning", {})
    )

    recovered_edges, audit_recovery = recover_antiparallel_cytokinesis_forks(
        smoothed_coords, pruned_edges, scale, descriptors, variant_cfg.get("cytokinesis_recovery", {})
    )

    cand = Lineage(smoothed_coords, recovered_edges)
    cand.validate()
    audit = {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
        "cytokinesis_recovery": audit_recovery,
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
        desc = f["descriptors"]
        stats = f["stats"]
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
        rec_cnt = audit["cytokinesis_recovery"].get("forks_recovered", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt}, rec={rec_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-024 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 024 on A1555 across {len(names)} development datasets...", flush=True)
    started = time.monotonic()

    results = {}
    for vname, vcfg in cfg["variants"].items():
        vdesc = vcfg["description"]
        print(f"Evaluating {vname}: {vdesc}...", flush=True)
        results[vname] = run_paired_evaluation(source, corpus, names, vname, vcfg)
        sc_delta = results[vname]["gate"]["score_delta"]
        is_passed = results[vname]["gate"]["passed"]
        print(f"  -> {vname} aggregate score delta: {sc_delta:+.6f}", flush=True)
        print(f"  -> {vname} passed: {is_passed}", flush=True)
        if not results[vname]["gate"]["passed"]:
            reasons = results[vname]["gate"]["reasons"]
            print(f"  -> Reasons: {reasons}", flush=True)

    passed_variants = [k for k, v in results.items() if v["gate"]["passed"]]
    if passed_variants:
        best_variant = max(passed_variants, key=lambda k: results[k]["gate"]["score_delta"])
        next_action = "eligible_for_confirmation_and_package"
    else:
        best_variant = None
        next_action = "rejected_no_kaggle_submission"

    overall = {
        "disposition": next_action,
        "best_variant": best_variant,
        "results": results,
        "seconds": time.monotonic() - started,
        "source_run": args.source_run,
        "code_hash": args.code_hash,
        "safety": {
            "isolated_execution": True,
            "no_mlflow": True,
            "no_gt_in_candidates": True,
        }
    }

    dump(run / "temporal_association.json", overall)
    return {
        "disposition": next_action,
        "best_variant": best_variant,
        "temporal_association_sha256": digest(run / "temporal_association.json"),
        "resource_profile": "cpu_evaluation_v1"
    }
