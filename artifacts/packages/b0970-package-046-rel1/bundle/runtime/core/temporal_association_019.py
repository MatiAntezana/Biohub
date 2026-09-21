"""019 Calibrated Axial Z Regularization and Conservative Gap Bridging and Conservative 1-Frame Gap Bridging.

Rama E + Rama D per docs/CAMPAIGN_019_PREDECLARATION.md.
Evaluates:
  Control: frozen baseline V19 reference graph.
  Variante 1: Conditional Z-Jitter Suppression (thresh=1.2 um, clamp=0.8 um, w_z=0.80, w_xy=0.0).
  Variante 2: Conditional Z-Jitter Suppression (thresh=1.5 um, clamp=1.0 um, w_z=0.85, w_xy=0.0).
  Variante 3: Conditional Z-Jitter Suppression (thresh=1.2 um, clamp=0.8 um) + Conservative 1-Frame Gap Bridging.

Topological Invariant: Reference forks and edges are 100% preserved.
Never prune reference forks to prevent division recall collapse.
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
from core.nextgen.contracts import Lineage, verify_file

ADDITIONS = {
    "core/temporal_association_018.py",
    "configs/temporal_association_019.json",
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
    return json.loads((ROOT / "configs/temporal_association_019.json").read_text())


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
        raise ValueError(f"019 requires source {cfg['source_run']}; got {folder.name}")
    report, corpus = audit_inputs(folder.parent, manifest)
    return corpus


def smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smoothing):
    """Smooths only anomalous axial Z jitter spikes along interior linear tracks.
    
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


def bridge_single_frame_gaps(coords, edges, scale, descriptors, cfg_gaps):
    """Bridges isolated single-frame gaps (t -> t+2) with 3D patch support.
    
    Inserts an interpolated node at t+1 and two edges (u -> w) and (w -> v).
    Conserves degree invariants: in_degree <= 1, out_degree <= 1.
    Respects spatial exclusion to prevent collisions with existing detections.
    """
    if not cfg_gaps.get("enabled", False):
        return coords.copy(), edges.copy(), {"gaps_bridged": 0, "nodes_added": 0, "edges_added": 0}

    max_dist_um = cfg_gaps.get("max_distance_um", 4.5)
    min_sim = cfg_gaps.get("min_similarity", 0.60)
    exclusion_um = cfg_gaps.get("spatial_exclusion_um", 2.0)
    max_bridges = cfg_gaps.get("max_bridges", 50)

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for u, v in edges:
        out_deg[int(u)] += 1
        in_deg[int(v)] += 1

    n_nodes = len(coords)
    # Mature endpoints only: in_deg == 1 (not a single isolated node)
    ends = [i for i in range(n_nodes) if out_deg[i] == 0 and in_deg[i] == 1]
    starts = [i for i in range(n_nodes) if in_deg[i] == 0 and out_deg[i] == 1]

    new_coords_list = [coords]
    new_edges_list = [edges]
    next_node_id = n_nodes
    gaps_bridged = 0

    max_t = int(coords[:, 0].max())
    by_t_coords = defaultdict(list)
    for i, c in enumerate(coords):
        by_t_coords[int(c[0])].append(c[1:] * scale)

    for t in range(max_t - 1):
        if gaps_bridged >= max_bridges:
            break
        ends_t = [i for i in ends if coords[i, 0] == t]
        starts_t2 = [i for i in starts if coords[i, 0] == t + 2]
        if not ends_t or not starts_t2:
            continue

        pos_ends = coords[ends_t, 1:] * scale
        pos_starts = coords[starts_t2, 1:] * scale

        tree_starts = cKDTree(pos_starts)
        tree_ends = cKDTree(pos_ends)

        existing_t1 = np.array(by_t_coords[t + 1]) if by_t_coords[t + 1] else None
        tree_existing_t1 = cKDTree(existing_t1) if existing_t1 is not None and len(existing_t1) > 0 else None

        for idx_u, u in enumerate(ends_t):
            if gaps_bridged >= max_bridges:
                break
            d, idx_v = tree_starts.query(pos_ends[idx_u])
            if d > max_dist_um:
                continue

            # Check mutual nearest neighbor
            d_back, idx_u_back = tree_ends.query(pos_starts[idx_v])
            if idx_u_back != idx_u:
                continue

            v = starts_t2[idx_v]
            # 3D Patch cosine similarity
            d_u = descriptors[u]
            d_v = descriptors[v]
            sim = float(np.dot(d_u, d_v) / (np.linalg.norm(d_u) * np.linalg.norm(d_v) + 1e-8))
            if sim < min_sim:
                continue

            # Candidate midpoint position in voxel space
            mid_vox = (coords[u, 1:] + coords[v, 1:]) / 2.0
            mid_phys = mid_vox * scale

            # Collision check with existing nodes at t+1
            if tree_existing_t1 is not None:
                d_near, _ = tree_existing_t1.query(mid_phys)
                if d_near < exclusion_um:
                    continue

            # Insert rescued node and edges
            new_node_coord = np.array([[float(t + 1), mid_vox[0], mid_vox[1], mid_vox[2]]], dtype=np.float64)
            new_edge_in = np.array([[u, next_node_id]], dtype=np.int64)
            new_edge_out = np.array([[next_node_id, v]], dtype=np.int64)

            new_coords_list.append(new_node_coord)
            new_edges_list.append(new_edge_in)
            new_edges_list.append(new_edge_out)

            by_t_coords[t + 1].append(mid_phys)
            tree_existing_t1 = cKDTree(np.array(by_t_coords[t + 1]))

            next_node_id += 1
            gaps_bridged += 1

    final_coords = np.vstack(new_coords_list)
    final_edges = np.vstack(new_edges_list)
    return final_coords, final_edges, {
        "gaps_bridged": gaps_bridged,
        "nodes_added": gaps_bridged,
        "edges_added": gaps_bridged * 2
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg):
    """Applies candidate transformations to the anchor reference graph."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    # Step 1: Conditional Z Jitter Suppression
    sm_cfg = variant_cfg.get("smoothing", {})
    coords, sm_audit = smooth_linear_tracks_conditional_z(
        coords, edges, scale, sm_cfg
    )

    # Step 2: Conservative 1-Frame Gap Bridging
    gaps_cfg = variant_cfg.get("gaps", {})
    coords, edges, gaps_audit = bridge_single_frame_gaps(
        coords, edges, scale, descriptors, gaps_cfg
    )

    return Lineage(coords, edges).validate(), {
        "smoothing": sm_audit,
        "gaps": gaps_audit
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
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        gap_cnt = audit["gaps"].get("gaps_bridged", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, gaps_bridged={gap_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-018 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 019 on A1555 across {len(names)} development datasets...", flush=True)
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
