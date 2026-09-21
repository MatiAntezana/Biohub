"""022 Frame-Shock Guarded and Density-Bounded Axial Regularization.

Campaign 022 per docs/AUTONOMOUS_CONTINUATION.md and docs/CAMPAIGN_022_PREDECLARATION.md:
Evaluates:
  Control: frozen baseline V19 reference graph (score = 0.901274).
  Variant 1: Pure Symmetric Control
             (reversal_only=True, symmetric_only=True, clamp_um=0.6, thresh=0.8 um, w_z=0.80).
             Established zero regressions on all datasets and +0.015235 on 44b6_ddf577ad.
  Variant 2: Guarded Axial Regularization (Conservative: max_frame_reversals=4, isolated_nn_thresh_um=6.5,
             clamp_um=0.6, thresh=0.8 um, w_z=0.80).
  Variant 3: Guarded Axial Regularization (Calibrated: max_frame_reversals=4, isolated_nn_thresh_um=5.5,
             clamp_um=0.6, thresh=0.75 um, w_z=0.85).

Mechanism (Frame-Shock Guarded and Density-Bounded Axial Regularization):
  1. Identifies directional reversal spikes in axial Z along interior linear tracks.
  2. Symmetric directional reversals (zp == zs) are unconditional single-frame optical
     bounce artifacts (established zero regressions across all datasets).
  3. Asymmetric directional reversals (zp != zs) are subject to two strict physical guards:
     - Collective Frame-Shock Guard: If > max_frame_reversals occur in the same frame t,
       the frame is flagged as a global microscope stage/focus jump and corrections are suppressed.
     - Spatial Density Collision Guard: If 3D nearest neighbor distance < isolated_nn_thresh_um,
       the node is in a crowded cluster where Z displacement could induce identity conflicts
       in bipartite matching. Corrections are suppressed.
  4. Lateral (Y, X) coordinates and graph topology (edges, nodes) are strictly untouched.
     Node count is strictly invariant at 235,674 rows.

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
}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "temporal_association_development_only_no_submission"


def config():
    return json.loads((ROOT / "configs/temporal_association_022.json").read_text())


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
        raise ValueError(f"022 requires source {src}; got {folder.name}")
    report, corpus = audit_inputs(folder.parent, manifest)
    return corpus


def smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smoothing):
    """Smooths anomalous axial Z jitter spikes with collective frame-shock and density guards.

    Parameters:
      - reversal_only (bool): If True, only smooths directional reversals in Z (dz1 * dz2 < 0).
      - symmetric_only (bool): If True, only smooths symmetric reversals (zp == zs).
      - max_frame_reversals (int): Frame-shock guard threshold. If asymmetric reversals at frame t
        exceed this count, frame t is treated as a global stage/focus shift and suppressed.
      - isolated_nn_thresh_um (float): Spatial density guard. Asymmetric reversals are suppressed
        if 3D nearest neighbor distance < isolated_nn_thresh_um.
      - jitter_threshold_um (float): Minimum deviation |zc - z_mid| * scale_z to qualify as a jitter spike.
      - max_correction_um (float): Maximum clamped axial displacement in um.
      - weight_z (float): Proportional correction factor towards z_mid.
      - weight_xy (float): Must be 0.0 (lateral coordinates strictly invariant).
    """
    if not cfg_smoothing.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0, "frame_shock_suppressed": 0, "density_suppressed": 0}

    reversal_only = cfg_smoothing.get("reversal_only", True)
    symmetric_only = cfg_smoothing.get("symmetric_only", False)
    max_frame_reversals = cfg_smoothing.get("max_frame_reversals", 4)
    isolated_nn_thresh_um = cfg_smoothing.get("isolated_nn_thresh_um", 6.0)
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
    by_t_pts = defaultdict(list)
    for i, c in enumerate(coords):
        t_val = int(c[0])
        by_t[t_val].add((float(c[1]), float(c[2]), float(c[3])))
        by_t_pts[t_val].append(c[1:] * scale)

    trees = {}
    for t_val, pts in by_t_pts.items():
        if len(pts) > 1:
            trees[t_val] = cKDTree(np.array(pts))

    # Pass 1: Identify all candidate spikes and compute per-frame asymmetric reversal counts
    reversal_cands = []
    asym_frame_counts = defaultdict(int)

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
            reversal_cands.append((nid, p, s, z_p, z_curr, z_s, z_mid, acc_z_um, is_symmetric))
            if not is_symmetric:
                asym_frame_counts[t_int] += 1

    # Pass 2: Apply Guards and Regularize
    spikes_suppressed = 0
    frame_shock_suppressed = 0
    density_suppressed = 0

    for nid, p, s, z_p, z_curr, z_s, z_mid, acc_z_um, is_symmetric in reversal_cands:
        t_int = int(coords[nid, 0])

        if not is_symmetric:
            # 1. Collective Frame-Shock Guard: stage/focus shift
            if asym_frame_counts[t_int] > max_frame_reversals:
                frame_shock_suppressed += 1
                continue

            # 2. Spatial Density Collision Guard: dense cluster bipartite protection
            if t_int in trees:
                dists, _ = trees[t_int].query(coords[nid, 1:] * scale, k=2)
                nn_dist = dists[1]
                if nn_dist < isolated_nn_thresh_um:
                    density_suppressed += 1
                    continue

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
        "frame_shock_suppressed": frame_shock_suppressed,
        "density_suppressed": density_suppressed,
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group="44b6"):
    """Applies candidate transformations to the anchor reference graph."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    sm_cfg = variant_cfg.get("smoothing", {})
    coords, sm_audit = smooth_linear_tracks_guarded_z(
        coords, edges, scale, sm_cfg
    )

    return Lineage(coords, edges).validate(), {
        "smoothing": sm_audit,
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
        fs_cnt = audit["smoothing"].get("frame_shock_suppressed", 0)
        den_cnt = audit["smoothing"].get("density_suppressed", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt} (shock_supp={fs_cnt}, dens_supp={den_cnt})", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-022 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 022 on A1555 across {len(names)} development datasets...", flush=True)
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
