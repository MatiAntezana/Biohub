"""023 Pure Symmetric Axial Regularization and Cytokinesis Geometric Pruning.

Campaign 023 per docs/AUTONOMOUS_CONTINUATION.md and docs/CAMPAIGN_023_PREDECLARATION.md:
Evaluates:
  Control: frozen baseline V19 reference graph (score = 0.901274).
  Variant 1: Pure Symmetric Regularizer + Conservative Mitosis Pruning (cos > 0.20)
  Variant 2: Pure Symmetric Regularizer + Zero-Cosine Hemispheric Pruning (cos > 0.00)
  Variant 3: Pure Symmetric Regularizer + Calibrated Cytokinesis Geometry Pruning (cos > -0.10, asymm > 8.0 um, sister < 3.0 um)

Mechanism (Pure Symmetric Axial Regularization + Cytokinesis Geometric Pruning):
  1. Identifies directional reversal spikes in axial Z along interior linear tracks.
     Symmetric directional reversals (zp == zs) are unconditional single-frame optical
     bounce artifacts (established zero regressions across all datasets).
  2. Spurious non-cytokinesis forks violating physical division geometry are pruned:
     - In eukaryotic cytokinesis, astral microtubules pull daughter chromatids to opposite poles.
     - Daughter divergence angle must be obtuse (cos(theta) < 0). Forks with cos(theta) >= 0
       represent false tracking merges/splits.
     - When pruning, the legitimate track continuation is identified via directional momentum
       alignment with the incoming trajectory (and patch similarity), and the intruder edge is removed.
  3. Node count is strictly invariant at 235,674 rows (well within [234,000, 242,000]).
  4. Degree invariants (in_degree <= 1, out_degree <= 2) are strictly preserved.

Topological & Metric Invariants:
  Strictly label-free during inference and candidate proposal. GT opened ONLY
  during final paired official evaluation.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from core.artifacts import ROOT, code_manifest, code_hash, digest, dump
from core import mitosis_counterfactual as official_eval
from core.protocol import assess
from core.nextgen import association as a, repair_ranker as r
from core.nextgen.contracts import Lineage, verify_file

ADDITIONS = {
    "core/temporal_association_023.py",
    "configs/temporal_association_023.json",
    "tests/test_temporal_association_023.py",
    "ops/audit_temporal_inputs_023.py",
    "core/temporal_association_022.py",
    "configs/temporal_association_022.json",
    "tests/test_temporal_association_022.py",
    "ops/audit_temporal_inputs_022.py",
    "core/temporal_association_021.py",
    "configs/temporal_association_021.json",
    "tests/test_temporal_association_021.py",
    "ops/audit_temporal_inputs_021.py",
    "core/temporal_association_020.py",
    "configs/temporal_association_020.json",
    "tests/test_temporal_association_020.py",
    "ops/audit_temporal_inputs_020.py",
    "core/temporal_association_019.py",
    "configs/temporal_association_019.json",
    "tests/test_temporal_association_019.py",
    "ops/audit_temporal_inputs_019.py",
    "core/temporal_association_018.py",
    "configs/temporal_association_018.json",
    "tests/test_temporal_association_018.py",
    "ops/audit_temporal_inputs_018.py",
    "core/temporal_association_017.py",
    "configs/temporal_association_017.json",
    "tests/test_temporal_association_017.py",
    "ops/audit_temporal_inputs_017.py",
    "core/temporal_association_016.py",
    "configs/temporal_association_016.json",
    "tests/test_temporal_association_016.py",
    "ops/audit_temporal_inputs_016.py",
    "core/stratified_ranking_015.py",
    "configs/event_ranker_015.json",
    "tests/test_stratified_ranking_015.py",
    "ops/audit_event_inputs_015.py",
    "core/coverage_audit.py",
    "configs/coverage_audit.json",
    "tests/test_coverage_audit.py",
    "ops/audit_coverage_inputs.py",
}
DRIVERS = {
    "ops/biohub.py",
    "ops/verify.py",
    "ops/bundle_build.py",
    "ops/kaggle.py",
    "core/remote_executor.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_023.json").read_text())


def audit_inputs(folder, manifest=None):
    """Verifies that b0970-selector-corpus-011 and source artifacts exist with exact hashes."""
    cfg = config()
    source_folder = Path(folder) / cfg["source_run"]
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Missing required source run folder: {source_folder}")

    for filename, expected_hash in cfg["source_artifacts"].items():
        filepath = source_folder / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Required source artifact missing: {filepath}")
        actual_hash = digest(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source artifact {filename} hash mismatch: {actual_hash} != {expected_hash}"
            )

    corpus_path = source_folder / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    files = list(source_folder.glob("*.evidence.npz"))
    if len(files) != 53:
        raise ValueError(f"Expected 53 evidence NPZ files, found {len(files)}")

    return dict(
        passed=True,
        evidence_npz=53,
        source_run=cfg["source_run"],
    ), corpus


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"]:
        src = cfg["source_run"]
        raise ValueError(f"023 requires source {src}; got {folder.name}")
    report, corpus = audit_inputs(folder.parent, manifest)
    return corpus


def smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smoothing):
    """Smooths anomalous axial Z jitter spikes on interior linear tracks."""
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
    """Prunes spurious non-cytokinesis forks that violate physical division geometry.

    Parameters:
      - enabled (bool): If False, returns edges unchanged.
      - cos_thresh (float): Maximum allowed divergence cosine. Forks with cos(theta) > cos_thresh are pruned.
      - max_asymm_um (float or None): If set, forks with distance asymmetry > max_asymm_um are pruned.
      - min_sister_um (float or None): If set, forks with sister distance < min_sister_um are pruned.
    """
    if not pr_cfg.get("enabled", False):
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.0)
    max_asymm_um = pr_cfg.get("max_asymm_um", None)
    min_sister_um = pr_cfg.get("min_sister_um", None)

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
        d_v1 = np.linalg.norm(v1)
        d_v2 = np.linalg.norm(v2)
        cos_th = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        asymm = float(abs(d_v1 - d_v2))
        sister_d = float(np.linalg.norm(v1 - v2))

        is_spurious = False
        if cos_th > cos_thresh:
            is_spurious = True
        elif max_asymm_um is not None and asymm > max_asymm_um:
            is_spurious = True
        elif min_sister_um is not None and sister_d < min_sister_um:
            is_spurious = True

        if is_spurious:
            # Determine keeper vs intruder
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

            # If descriptors available, add morphology similarity
            if descriptors is not None and len(descriptors) > max(m, d1, d2):
                d_m = descriptors[m]
                norm_m = np.linalg.norm(d_m)
                if norm_m > 0:
                    sim1 = float(np.dot(d_m, descriptors[d1]) / max(norm_m * np.linalg.norm(descriptors[d1]), 1e-6))
                    sim2 = float(np.dot(d_m, descriptors[d2]) / max(norm_m * np.linalg.norm(descriptors[d2]), 1e-6))
                    score1 += 0.3 * sim1
                    score2 += 0.3 * sim2

            if score1 >= score2:
                # d1 is keeper, d2 is intruder
                edges_to_remove.add((m, d2))
            else:
                # d2 is keeper, d1 is intruder
                edges_to_remove.add((m, d1))

    retained_edges = np.array([e for e in edges if (int(e[0]), int(e[1])) not in edges_to_remove], dtype=np.int64)
    return retained_edges, {
        "forks_inspected": len(divs),
        "forks_pruned": len(edges_to_remove),
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group="44b6"):
    """Applies candidate transformations to the anchor reference graph."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    # 1. Axial regularizer
    sm_cfg = variant_cfg.get("smoothing", {})
    coords, sm_audit = smooth_linear_tracks_guarded_z(
        coords, edges, scale, sm_cfg
    )

    # 2. Cytokinesis geometric fork pruning
    pr_cfg = variant_cfg.get("fork_pruning", {})
    edges, pr_audit = prune_spurious_cytokinesis_forks(
        coords, edges, scale, descriptors, pr_cfg
    )

    return Lineage(coords, edges).validate(), {
        "smoothing": sm_audit,
        "fork_pruning": pr_audit,
    }


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
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, forks_pruned={pr_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-023 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 023 on A1555 across {len(names)} development datasets...", flush=True)
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
        "source_run": source.name,
        "code_hash": code_hash(),
        "safety": {
            "uses_gt_for_edits": False,
            "graph_exported": False,
            "tracking_evaluated": True,
            "kaggle_submission_allowed": bool(passed_variants),
            "promoted": False
        }
    }
    dump(run / "temporal_association.json", overall)
    return {"disposition": next_action, "best_variant": best_variant,
            "temporal_association_sha256": digest(run / "temporal_association.json")}
