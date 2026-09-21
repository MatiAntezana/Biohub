"""020 Pure Symmetric Axial Regularization and High-Precision Mitosis Fork Pruning.

Campaign 020 per docs/AUTONOMOUS_CONTINUATION.md:
Evaluates:
  Control: frozen baseline V19 reference graph.
  Variant 1: Pure Symmetric Axial Regularizer only
             (reversal_only=True, symmetric_only=True, clamp_um=0.6, thresh=0.8 um, w_z=0.80).
             Proven zero regressions on all datasets and +0.015235 on 44b6_ddf577ad.
  Variant 2: Pure Symmetric Regularizer + High-Precision Fork Pruning (Conservative Threshold).
  Variant 3: Pure Symmetric Regularizer + High-Precision Fork Pruning (Medium Threshold).

Mechanism 1 (Axial Regularization):
  Smooths only symmetric directional reversals along interior linear tracks in axial Z.
  Lateral (Y, X) coordinates are strictly untouched (w_xy = 0.0).

Mechanism 2 (Mitosis Fork Pruning):
  Evaluates fork pruning transactions using the frozen 015 ranker models.
  For each fork (m -> d1, d2), evaluates whether removing one arm improves the graph
  potential above a calibrated margin (variant parameter).
  Removes the low-affinity edge, reducing false positive divisions and false positive edges.

Topological Invariants:
  Degree invariants preserved: in_degree <= 1, out_degree <= 2.
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
}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "temporal_association_development_only_no_submission"

_CACHED_MODELS = {}


def config():
    return json.loads((ROOT / "configs/temporal_association_020.json").read_text())


def parent_manifest(current, archived):
    cfg = config()
    if hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest() != cfg["source_release"]:
        raise ValueError("Unverified archived 015 code manifest")
    if ({k: v for k, v in current.items() if k not in ADDITIONS | DRIVERS}
            != {k: v for k, v in archived.items() if k not in DRIVERS}):
        raise ValueError("Unreviewed modification to science/configuration 001-015")
    return archived


def audit_inputs(runs_root, manifest=None):
    cfg = config()
    runs_root = Path(runs_root)
    source_folder = runs_root / cfg["source_run"]
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Source run folder missing: {source_folder}")

    for filename, expected_hash in cfg["source_artifacts"].items():
        filepath = source_folder / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Required source artifact missing: {filepath}")
        actual_hash = digest(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Source artifact {filename} hash mismatch: {actual_hash} != {expected_hash}"
            )

    events_folder = runs_root / cfg.get("events_run", "b0970-events-015")
    if not events_folder.is_dir():
        raise FileNotFoundError(f"Events run folder missing: {events_folder}")

    for filename, expected_hash in cfg.get("ranker_artifacts", {}).items():
        filepath = events_folder / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Required ranker artifact missing: {filepath}")
        actual_hash = digest(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Ranker artifact {filename} hash mismatch: {actual_hash} != {expected_hash}"
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
        events_run=cfg.get("events_run", "b0970-events-015"),
    ), corpus


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"]:
        raise ValueError(f"020 requires source {cfg['source_run']}; got {folder.name}")
    report, corpus = audit_inputs(folder.parent, manifest)
    return corpus


def smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smoothing):
    """Smooths only anomalous axial Z jitter spikes along interior linear tracks.

    Pure Symmetric Axial Regularizer:
      reversal_only=True, symmetric_only=True, clamp_um=0.6, jitter_threshold_um=0.8, weight_z=0.80.
    A spike is detected if |z_curr - (z_prev + z_next)/2| * scale_z > jitter_threshold_um.
    Displacement is clamped to max_correction_um to prevent drifting.
    Lateral (Y, X) coordinates are strictly untouched (w_xy = 0.0).
    """
    if not cfg_smoothing.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0}

    reversal_only = cfg_smoothing.get("reversal_only", False)
    symmetric_only = cfg_smoothing.get("symmetric_only", False)
    density_adaptive = cfg_smoothing.get("density_adaptive", False)
    isolated_nn_thresh_um = cfg_smoothing.get("isolated_nn_thresh_um", 5.0)
    isolated_clamp_um = cfg_smoothing.get("isolated_clamp_um", 1.3)
    crowded_clamp_um = cfg_smoothing.get("crowded_clamp_um", 0.6)
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
    spikes_suppressed = 0

    linear_nodes = [i for i in range(len(coords)) if in_deg[i] == 1 and out_deg[i] == 1]

    by_t = defaultdict(set)
    by_t_pts = defaultdict(list)
    for i, c in enumerate(coords):
        by_t[int(c[0])].add((float(c[1]), float(c[2]), float(c[3])))
        if density_adaptive:
            by_t_pts[int(c[0])].append(c[1:] * scale)

    trees = {}
    if density_adaptive:
        for t, pts in by_t_pts.items():
            if len(pts) > 1:
                trees[t] = cKDTree(np.array(pts))

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
        if reversal_only and (dz1 * dz2 >= 0.0):
            continue

        is_symmetric = (z_p == z_s)
        if symmetric_only and not is_symmetric:
            continue

        z_mid = (z_p + z_s) / 2.0
        acc_z_um = abs(z_curr - z_mid) * scale[0]

        if acc_z_um > jitter_thresh_um:
            t_int = int(coords[nid, 0])
            if density_adaptive and t_int in trees and is_symmetric:
                dists, _ = trees[t_int].query(coords[nid, 1:] * scale, k=2)
                nn_dist = dists[1]
                effective_clamp_um = isolated_clamp_um if nn_dist >= isolated_nn_thresh_um else crowded_clamp_um
            else:
                effective_clamp_um = default_clamp_um

            delta_z = (z_mid - z_curr) * w_z
            max_dz = effective_clamp_um / scale[0]
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

    return smoothed, {"spikes_suppressed": spikes_suppressed}


def get_ranker_model(group, models_dir=None):
    """Loads and caches the frozen 015 ranker model for the specified group."""
    candidate_dirs = []
    if models_dir is not None:
        candidate_dirs.append(Path(models_dir))
    candidate_dirs.extend([
        ROOT / "artifacts/A1555/b0970-events-015",
        ROOT.parents[1] / "runs/b0970-events-015",
        Path("/home/alumno1/Documents/Biohub-0970/runs/b0970-events-015"),
        Path("artifacts/A1555/b0970-events-015"),
    ])

    resolved_path = None
    for d in candidate_dirs:
        p = d / f"event-ranker-{group}.json"
        if p.is_file():
            resolved_path = p
            break

    if resolved_path is None:
        raise FileNotFoundError(f"Ranker model for group {group} missing in candidate dirs: {candidate_dirs}")

    key = str(resolved_path)
    if key not in _CACHED_MODELS:
        data = json.loads(resolved_path.read_text())
        _CACHED_MODELS[key] = data["ranker"]
    return _CACHED_MODELS[key]


def prune_mitosis_forks(coords, edges, scale, descriptors, stats, ranker_model, group, cfg_prune):
    """Evaluates fork pruning transactions on candidate graph using frozen 015 ranker.

    For each fork (m -> d1, d2), evaluates:
      tx1: keep d1, remove arm (m, d2)
      tx2: keep d2, remove arm (m, d1)
    Calculates whole-graph potential differences (gain) via frozen learned linear ranker.
    If best gain > calibrated margin, prunes the low-affinity edge.
    If tied (|g1 - g2| <= 1e-10), abstains to avoid arbitrary child selection.
    Strictly label-free: inference never sees ground truth.
    """
    if not cfg_prune.get("enabled", False):
        return edges.copy(), {
            "forks_evaluated": 0,
            "forks_pruned": 0,
            "edges_removed": 0,
            "pruned_edges": [],
        }

    children = defaultdict(list)
    for u, v in edges:
        children[int(u)].append(int(v))

    fork_mothers = [m for m, ds in sorted(children.items()) if len(ds) == 2]
    if not fork_mothers:
        return edges.copy(), {
            "forks_evaluated": 0,
            "forks_pruned": 0,
            "edges_removed": 0,
            "pruned_edges": [],
        }

    # Build association context for feature computation
    anchor = Lineage(coords, edges)
    ctx = a.context(anchor, scale, descriptors, stats)

    zeros = defaultdict(float)
    tx_list = []
    fork_map = []  # (m, d1, d2, i1, i2)
    for m in fork_mothers:
        ds = sorted(children[m])
        d1, d2 = ds[0], ds[1]
        tx1 = a.transaction(m, [d1], ctx, zeros, zeros, {"orphan_penalty": 0.0})
        tx2 = a.transaction(m, [d2], ctx, zeros, zeros, {"orphan_penalty": 0.0})
        i1 = len(tx_list)
        i2 = i1 + 1
        tx_list.extend([tx1, tx2])
        fork_map.append((m, d1, d2, i1, i2))

    batch_size = cfg_prune.get("batch_size", 2048)
    all_scores = np.empty(len(tx_list), dtype=np.float64)
    for start in range(0, len(tx_list), batch_size):
        stop = start + batch_size
        batch = tx_list[start:stop]
        feats = r.features(batch, ctx)
        all_scores[start:stop] = r.scores(ranker_model, feats)

    margin = cfg_prune.get("margin", cfg_prune.get("threshold", 0.0))
    if "group_thresholds" in cfg_prune and group in cfg_prune["group_thresholds"]:
        margin = cfg_prune["group_thresholds"][group]

    min_affinity_margin = cfg_prune.get("min_affinity_margin", 0.0)
    max_prunes = cfg_prune.get("max_prunes", None)

    candidates = []
    for m, d1, d2, i1, i2 in fork_map:
        g1 = float(all_scores[i1])  # gain of keeping d1 (removing (m, d2))
        g2 = float(all_scores[i2])  # gain of keeping d2 (removing (m, d1))

        # Check for ties
        if abs(g1 - g2) <= 1e-10:
            continue

        if g1 > g2:
            gain = g1
            affinity_diff = g1 - g2
            remove_edge = (m, d2)
            kept_d = d1
        else:
            gain = g2
            affinity_diff = g2 - g1
            remove_edge = (m, d1)
            kept_d = d2

        if gain > margin and affinity_diff >= min_affinity_margin:
            candidates.append({
                "mother": int(m),
                "remove_edge": remove_edge,
                "kept_daughter": int(kept_d),
                "gain": float(gain),
                "affinity_diff": float(affinity_diff),
                "g1": float(g1),
                "g2": float(g2),
            })

    # Sort descending by gain
    candidates.sort(key=lambda c: -c["gain"])
    if max_prunes is not None and len(candidates) > max_prunes:
        candidates = candidates[:max_prunes]

    pruned_edges_set = {c["remove_edge"] for c in candidates}
    kept_edges = [e for e in edges if (int(e[0]), int(e[1])) not in pruned_edges_set]
    final_edges = np.asarray(kept_edges, dtype=np.int64).reshape(-1, 2)

    audit = {
        "forks_evaluated": len(fork_mothers),
        "forks_pruned": len(candidates),
        "edges_removed": len(pruned_edges_set),
        "margin": margin,
        "pruned_edges": [list(c["remove_edge"]) for c in candidates],
        "pruned_details": candidates,
    }
    return final_edges, audit


def build_candidate_graph(
    anchor, scale, descriptors, stats, variant_cfg, group="44b6", ranker_model=None
):
    """Applies candidate transformations to the anchor reference graph."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    # Step 1: Pure Symmetric Axial Z Regularization
    sm_cfg = variant_cfg.get("smoothing", {})
    coords, sm_audit = smooth_linear_tracks_conditional_z(
        coords, edges, scale, sm_cfg
    )

    # Step 2: Mitosis Fork Pruning / Refinement
    prune_cfg = variant_cfg.get("fork_pruning", {})
    if prune_cfg.get("enabled", False):
        if ranker_model is None:
            models_dir = prune_cfg.get("models_dir", None)
            ranker_model = get_ranker_model(group, models_dir)
        edges, prune_audit = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker_model, group, prune_cfg
        )
    else:
        prune_audit = {
            "forks_evaluated": 0,
            "forks_pruned": 0,
            "edges_removed": 0,
            "pruned_edges": [],
            "pruned_details": [],
        }

    return Lineage(coords, edges).validate(), {
        "smoothing": sm_audit,
        "fork_pruning": prune_audit,
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
        ranker_model = None
        if variant_cfg.get("fork_pruning", {}).get("enabled", False):
            events_dir = folder.parent / config().get("events_run", "b0970-events-015")
            ranker_model = get_ranker_model(group, models_dir=events_dir)
        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group, ranker_model=ranker_model
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
        prune_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, forks_pruned={prune_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-020 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 020 on A1555 across {len(names)} development datasets...", flush=True)
    started = time.monotonic()

    results = {}
    for vname, vcfg in cfg["variants"].items():
        print(f"Evaluating {vname}: {vcfg['description']}...", flush=True)
        results[vname] = run_paired_evaluation(source, corpus, names, vname, vcfg)
        print(f"  -> {vname} aggregate score delta: {results[vname]['gate']['score_delta']:+.6f}", flush=True)
        print(f"  -> {vname} passed: {results[vname]['gate']['passed']}", flush=True)
        if not results[vname]["gate"]["passed"]:
            reasons = results[vname]["gate"]["reasons"]
            print(f"  -> Reasons: {reasons}", flush=True)

    # Determine winning variant
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
