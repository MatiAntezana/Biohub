"""017 Invariant Mitotic Fork Pruning and Conditional Axial Jitter Suppression.

Rama E / Mitosis Invariant Pruning per docs/CAMPAIGN_017_PREDECLARATION.md.
Evaluates:
  Control: frozen baseline V19 reference graph.
  Variante 1: Biological Invariant Fork Pruning (3D patch correlation + sister symmetry).
  Variante 2: Biological Fork Pruning + Conditional Axial Z Jitter Suppression.

Strictly label-free during inference and candidate proposal. GT opened ONLY
during final paired official evaluation.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from core.artifacts import ROOT, code_manifest, code_hash, digest, dump
from core import mitosis_counterfactual as official_eval
from core import mitosis_selector as selector
from core.nextgen.contracts import Lineage, verify_file

ADDITIONS = {
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


def config():
    return json.loads((ROOT / "configs/temporal_association_017.json").read_text())


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

    corpus_path = source_folder / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    files = list(source_folder.glob("*.evidence.npz"))
    if len(files) != 53:
        raise ValueError(f"Expected 53 evidence NPZ files, found {len(files)}")

    return dict(passed=True, evidence_npz=53, source_run=cfg["source_run"]), corpus


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"]:
        raise ValueError(f"017 requires source {cfg['source_run']}; got {folder.name}")
    report, corpus = audit_inputs(folder.parent, manifest)
    return corpus


def prune_spurious_forks(coords, edges, scale, descriptors, stats, cfg_pruning):
    """Evaluates 3D biological invariants for forks and prunes imposter branches.
    
    A genuine biological division satisfies:
    1. Mother-daughter patch resemblance: min(dot(m, d1), dot(m, d2)) >= min_mother_dot
    2. Sister-sister patch resemblance: dot(d1, d2) >= min_sister_dot
    3. Distance bounds: max(dist(m, d1), dist(m, d2)) <= max_dist_um
    4. Sister separation bound: dist(d1, d2) <= max_sister_sep_um
    5. Symmetry bound: |dist(m, d1) - dist(m, d2)| <= max_dist_diff_um
    
    Forks violating invariants have their imposter edge pruned, preserving the true cell.
    Conserves node cardinality: Delta nodes = 0.
    Reduces edge and division false positives.
    """
    if not cfg_pruning.get("enabled", False):
        return edges.copy(), {"forks_evaluated": 0, "forks_certified": 0, "forks_pruned": 0}

    min_mother_dot = cfg_pruning.get("min_mother_dot", 0.45)
    min_sister_dot = cfg_pruning.get("min_sister_dot", 0.35)
    max_dist_um = cfg_pruning.get("max_dist_um", 4.8)
    max_sister_sep_um = cfg_pruning.get("max_sister_sep_um", 7.0)
    max_dist_diff_um = cfg_pruning.get("max_dist_diff_um", 2.5)

    out_edges = defaultdict(list)
    for u, v in edges:
        out_edges[int(u)].append(int(v))

    edges_to_remove = set()
    forks_evaluated = 0
    forks_certified = 0
    forks_pruned = 0

    for m, daughters in sorted(out_edges.items()):
        if len(daughters) != 2:
            continue

        forks_evaluated += 1
        d1, d2 = daughters[0], daughters[1]

        # Physical distances
        dist1 = float(np.linalg.norm((coords[d1, 1:] - coords[m, 1:]) * scale))
        dist2 = float(np.linalg.norm((coords[d2, 1:] - coords[m, 1:]) * scale))
        sep = float(np.linalg.norm((coords[d1, 1:] - coords[d2, 1:]) * scale))
        diff = abs(dist1 - dist2)

        # 3D Patch correlations (unit normalized descriptors)
        dot1 = float(np.dot(descriptors[m], descriptors[d1]))
        dot2 = float(np.dot(descriptors[m], descriptors[d2]))
        sister_dot = float(np.dot(descriptors[d1], descriptors[d2]))

        # Invariant checks
        is_symmetric_division = (
            dot1 >= min_mother_dot and
            dot2 >= min_mother_dot and
            sister_dot >= min_sister_dot and
            dist1 <= max_dist_um and
            dist2 <= max_dist_um and
            sep <= max_sister_sep_um and
            diff <= max_dist_diff_um
        )

        if is_symmetric_division:
            forks_certified += 1
        else:
            # Fork fails biological invariants: identify and prune the imposter edge
            # Score each branch: combination of appearance fidelity and proximity
            score1 = dot1 - 0.05 * (dist1 / 5.0)
            score2 = dot2 - 0.05 * (dist2 / 5.0)

            if score1 >= score2:
                # Keep d1, prune imposter d2
                edges_to_remove.add((m, d2))
            else:
                # Keep d2, prune imposter d1
                edges_to_remove.add((m, d1))
            forks_pruned += 1

    pruned_edges = np.array([e for e in edges if tuple(e) not in edges_to_remove], dtype=np.int64)
    audit = {
        "forks_evaluated": forks_evaluated,
        "forks_certified": forks_certified,
        "forks_pruned": forks_pruned,
        "edges_pruned": len(edges_to_remove),
    }
    return pruned_edges, audit


def smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smoothing):
    """Conditional Z-axis jitter suppression on linear track interiors.
    
    Only modifies Z when local acceleration |z_0 - (z_{-1} + z_1)/2| exceeds
    a strict jitter threshold (e.g. 2.5 um). Clamps maximum displacement to
    never exceed max_correction_um (1.5 um). Never touches lateral (Y, X) coordinates.
    Preserves curve fidelity and coordinate uniqueness.
    """
    if not cfg_smoothing.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0}

    jitter_thresh_um = cfg_smoothing.get("jitter_threshold_um", 2.5)
    max_corr_um = cfg_smoothing.get("max_correction_um", 1.5)
    w_z = cfg_smoothing.get("weight_z", 0.85)

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
    for i, c in enumerate(coords):
        by_t[int(c[0])].add((float(c[1]), float(c[2]), float(c[3])))

    for nid in linear_nodes:
        p = pred[nid]
        s = succ[nid]
        # Must be consecutive time frames: t-1, t, t+1
        if coords[nid, 0] != coords[p, 0] + 1 or coords[s, 0] != coords[nid, 0] + 1:
            continue

        z_p = coords[p, 1]
        z_curr = coords[nid, 1]
        z_s = coords[s, 1]

        z_mid = (z_p + z_s) / 2.0
        acc_z_um = abs(z_curr - z_mid) * scale[0]

        if acc_z_um > jitter_thresh_um:
            # Anomalous Z spike detected: apply damped regularisation
            delta_z = (z_mid - z_curr) * w_z
            max_dz = max_corr_um / scale[0]
            delta_z_clamped = np.clip(delta_z, -max_dz, max_dz)
            new_z = max(0.0, float(z_curr + delta_z_clamped))

            t_int = int(coords[nid, 0])
            orig_key = (float(z_curr), float(coords[nid, 2]), float(coords[nid, 3]))
            cand_key = (new_z, float(coords[nid, 2]), float(coords[nid, 3]))

            if cand_key != orig_key and cand_key in by_t[t_int]:
                continue

            by_t[t_int].discard(orig_key)
            by_t[t_int].add(cand_key)
            smoothed[nid, 1] = new_z
            spikes_suppressed += 1

    return smoothed, {"spikes_suppressed": spikes_suppressed}


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg):
    """Applies candidate transformations to the anchor reference graph."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    # Step 1: Biological fork pruning
    fp_cfg = variant_cfg.get("fork_pruning", {})
    edges, fp_audit = prune_spurious_forks(
        coords, edges, scale, descriptors, stats, fp_cfg
    )

    # Step 2: Conditional Z jitter suppression
    sm_cfg = variant_cfg.get("smoothing", {})
    coords, sm_audit = smooth_linear_tracks_conditional_z(
        coords, edges, scale, sm_cfg
    )

    cand_lineage = Lineage(coords, edges)
    cand_lineage.validate()
    return cand_lineage, {"fork_pruning": fp_audit, "smoothing": sm_audit}


def run_paired_evaluation(folder, corpus, names, variant_name, variant_cfg):
    """Runs label-free candidate generation followed by official GT evaluation."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    metric = official_eval.official_metrics()
    scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

    refs = {}
    cands = {}
    audits = {}

    for name in names:
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
        cand_graph, audit = build_candidate_graph(anchor, scale, desc, stats, variant_cfg)
        audits[name] = audit

        # 3. Open GT for official metric evaluation ONLY
        gt_path = Path(corpus["data_root"]) / (name + ".geff")
        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            raise ValueError(f"Valid estimated_number_of_nodes required for {name}")

        row = {"name": name, "group": name.split("_", 1)[0], "split": "development",
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
        pruned_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, pruned_forks={pruned_cnt}, z_spikes={z_cnt}", flush=True)

    evaluation = selector.evaluate_panel(refs, cands)
    gate = selector.assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-017 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 017 on A1555 across {len(names)} development datasets...", flush=True)
    started = time.monotonic()

    results = {}
    for vname, vcfg in cfg["variants"].items():
        print(f"Evaluating {vname}: {vcfg['description']}...", flush=True)
        results[vname] = run_paired_evaluation(source, corpus, names, vname, vcfg)
        print(f"  -> {vname} aggregate score delta: {results[vname]['gate']['score_delta']:+.6f}", flush=True)
        print(f"  -> {vname} passed: {results[vname]['gate']['passed']}", flush=True)
        if not results[vname]["gate"]["passed"]:
            print(f"  -> Reasons: {results[vname]['gate']['reasons']}", flush=True)

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
