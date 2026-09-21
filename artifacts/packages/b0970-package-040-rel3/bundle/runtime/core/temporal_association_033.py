"""033: Drift-Compensated Cytokinesis-directed Mitosis Reassignment (DC-CMR).

Builds on Campaign 032's DDKS-SPI (smoothing + fork pruning) and adds a
post-backbone greedy reassignment pass.  When a single-daughter mother m
at frame t has a candidate second daughter d2 at t+1 currently assigned
to a linear competitor c, DC-CMR reassigns d2 from c to m if:

  1. Rest-frame cytokinesis angle cos(theta*) <= cos_max  (antiparallel)
  2. Sister distance |d1 - d2| in [sister_min, sister_max] um
  3. Distance ratio min(|d1-m|, |d2-m|) / max(...) >= ratio_min
  4. Raw detector p(m -> d2) >= p_candidate_min   (edge existed)
  5. Competitor c has out_deg <= competitor_max_out_deg

Rest-frame drift is computed as per-frame global median displacement
vec{v}_global(t) over all active forward edges, eliminating axial
discretization artefacts from coarse z-spacing.

The reassignment is edge-count neutral (-1 removed + 1 added = 0).
Node cardinality is unchanged.  In-degree <= 1 and out-degree <= 2
are maintained.  Label-free candidate generation (GT never accessed).
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core import mitosis_counterfactual as official_eval
from core.nextgen.association import Lineage
from core.nextgen.contracts import load_reference, verify_file
from core.protocol import assess

# Re-use 032's smoothing and fork-pruning functions identically
from core.temporal_association_032 import (
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
)

ADDITIONS = {
    "core/temporal_association_033.py",
    "configs/temporal_association_033.json",
    "tests/test_temporal_association_033.py",
    "ops/audit_temporal_inputs_033.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_033.json").read_text())


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"] and folder.name != "b0970-prepare-005":
        exp = cfg["source_run"]
        raise ValueError(f"Expected source run {exp} or b0970-prepare-005, got {folder.name}")
    receipt_path = folder / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing receipt in source run: {folder}")
    receipt = json.loads(receipt_path.read_text())
    task = receipt.get("task")
    status = receipt.get("status")
    if status != "completed":
        raise ValueError(f"Source run {folder.name} status is {status}, expected completed")

    if task == "prepare":
        plan_path = folder / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Plan missing: {plan_path}")
        plan = json.loads(plan_path.read_text())
        if "data_root" not in plan or not Path(plan["data_root"]).is_dir():
            local_alt = folder / "cache-confirmation/raw"
            if local_alt.is_dir():
                plan["data_root"] = str(local_alt)
            else:
                a1688_alt = folder.parent.parent / "A1688" / folder.name / "cache-confirmation/raw"
                if a1688_alt.is_dir():
                    plan["data_root"] = str(a1688_alt)
        return plan

    elif task == "prepare-selector":
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
            "data_root": str(folder.parent.parent / "A1688/b0970-prepare-004/cache-development/raw"),
        }
    else:
        raise ValueError(f"Unsupported source task: {task}")


def audit_inputs(runs_dir):
    runs_dir = Path(runs_dir)
    cfg = config()
    source = runs_dir / cfg["source_run"]
    if not source.is_dir():
        local_alt = ROOT / "artifacts/A1555" / cfg["source_run"]
        if local_alt.is_dir():
            source = local_alt
        else:
            raise FileNotFoundError(f"Source run directory missing: {source}")
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    npz_stats = {}
    for name in names:
        path = source / (name + ".evidence.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Evidence missing: {path}")
        with np.load(path) as data:
            coords = data["coords"]
            edges = data["reference_edges"]
            scale = data["scale"]
            desc = data["descriptors"] if "descriptors" in data else None
            npz_stats[name] = {
                "nodes": int(len(coords)),
                "edges": int(len(edges)),
                "has_descriptors": desc is not None,
                "group": name.split("_", 1)[0],
            }
    # Also verify that candidate NPZ files are available for DC-CMR
    candidates_available = {}
    for name in names:
        cand_path = ROOT / "artifacts/A1688/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz")
        candidates_available[name] = cand_path.is_file()
    return {
        "status": "audit_passed",
        "source_run": cfg["source_run"],
        "development_datasets": len(names),
        "stats": npz_stats,
        "candidates_available": candidates_available,
    }, corpus


# ---------------------------------------------------------------------------
# DC-CMR: Drift-Compensated Cytokinesis-directed Mitosis Reassignment
# ---------------------------------------------------------------------------

def compute_global_drift(coords, edges, scale):
    """Per-frame global median drift vector v_global(t) in physical units (um).

    For each frame t, computes the median displacement of all forward edges
    (t -> t+1).  Returns dict mapping int(t) -> np.array([dz, dy, dx]) in um.
    """
    t_sources = coords[edges[:, 0], 0].astype(int)
    drifts = {}
    for t in np.unique(t_sources):
        mask = t_sources == t
        sub = edges[mask]
        t_targets = coords[sub[:, 1], 0].astype(int)
        fwd = t_targets == t + 1
        if np.any(fwd):
            disps = (coords[sub[fwd, 1], 1:] - coords[sub[fwd, 0], 1:]) * scale
            drifts[int(t)] = np.median(disps, axis=0)
        else:
            drifts[int(t)] = np.zeros(3)
    return drifts


def _build_anchor_to_cand_map(coords, scale, cand_npz):
    """Maps every anchor node to the nearest candidate-graph node using KDTree.

    Returns anchor_to_cand (int array) and cand_edges probability lookup dict.
    """
    from scipy.spatial import cKDTree

    cand_coords = cand_npz["coords"]
    cand_by_t = {}
    for t_val in np.unique(cand_coords[:, 0].astype(int)):
        mask = cand_coords[:, 0] == t_val
        idxs = np.flatnonzero(mask)
        tree = cKDTree(cand_coords[idxs, 1:] * scale)
        cand_by_t[int(t_val)] = (idxs, tree)

    anchor_to_cand = np.full(len(coords), -1, dtype=int)
    for t_val in np.unique(coords[:, 0].astype(int)):
        t_val = int(t_val)
        if t_val not in cand_by_t:
            continue
        mask_a = coords[:, 0].astype(int) == t_val
        idxs_a = np.flatnonzero(mask_a)
        idxs_c, tree = cand_by_t[t_val]
        dists, local_idxs = tree.query(coords[idxs_a, 1:] * scale)
        anchor_to_cand[idxs_a] = idxs_c[local_idxs]

    # Build probability lookup: (raw_u, raw_v) -> p
    cand_edges = cand_npz["edges"]
    prob_map = {}
    for row in cand_edges:
        prob_map[(int(row[0]), int(row[1]))] = float(row[2])

    return anchor_to_cand, prob_map


def apply_dc_cmr(coords, edges, scale, reassign_cfg, cand_npz=None):
    """Greedy rest-frame cytokinesis-directed mitosis reassignment.

    For each single-daughter mother m with child d1, searches for a candidate
    second daughter d2 currently owned by competitor c, and reassigns d2 from
    c to m if cytokinesis geometry and detector probability pass thresholds.

    Returns (new_edges, audit_dict).
    """
    if not reassign_cfg.get("enabled", False):
        return edges, {"reassignments": 0, "candidates_scanned": 0}

    cos_max = reassign_cfg.get("cos_max", -0.30)
    sister_min_um = reassign_cfg.get("sister_distance_min_um", 3.0)
    sister_max_um = reassign_cfg.get("sister_distance_max_um", 16.0)
    ratio_min = reassign_cfg.get("distance_ratio_min", 0.15)
    p_cand_min = reassign_cfg.get("p_candidate_min", 0.45)
    comp_max_out = reassign_cfg.get("competitor_max_out_deg", 1)
    use_drift = reassign_cfg.get("global_drift_compensation", True)

    # Build graph topology
    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    # Compute global drift
    if use_drift:
        drifts = compute_global_drift(coords, edges, scale)
    else:
        drifts = {}

    # Map anchor nodes to candidate detector nodes for probability lookup
    anchor_to_cand = None
    prob_map = {}
    if cand_npz is not None:
        anchor_to_cand, prob_map = _build_anchor_to_cand_map(coords, scale, cand_npz)

    # Pre-index parents by time of d2: t -> list of (d2, c)
    parents_by_t = defaultdict(list)
    for d2, c in parents.items():
        parents_by_t[int(coords[d2, 0])].append((d2, c))

    # Identify single-daughter mothers at each frame
    mothers = [m for m, ch in children.items() if len(ch) == 1]
    candidates_scanned = 0

    # Collect all candidate reassignments, then greedily select
    candidate_reassignments = []

    for m in mothers:
        d1 = children[m][0]
        t_m = int(coords[m, 0])
        if int(coords[d1, 0]) != t_m + 1:
            continue

        r_m = coords[m, 1:] * scale
        r_d1 = coords[d1, 1:] * scale
        v_drift = drifts.get(t_m, np.zeros(3))

        # Rest-frame displacement to d1
        v1_star = (r_d1 - r_m) - v_drift
        norm_v1 = float(np.linalg.norm(v1_star))
        if norm_v1 < 1e-6:
            continue

        # Search for candidate d2 nodes at t_m+1
        for d2, c in parents_by_t.get(t_m + 1, ()):
            if d2 == d1:
                continue
            if c == m:
                continue
            if int(coords[c, 0]) != t_m:
                continue
            if len(children[c]) > comp_max_out:
                continue

            candidates_scanned += 1
            r_d2 = coords[d2, 1:] * scale

            # Sister distance check
            sister_dist = float(np.linalg.norm(r_d1 - r_d2))
            if sister_dist < sister_min_um or sister_dist > sister_max_um:
                continue

            # Distance from mother to d2
            v2_star = (r_d2 - r_m) - v_drift
            norm_v2 = float(np.linalg.norm(v2_star))
            if norm_v2 < 1e-6:
                continue

            # Distance ratio
            ratio = min(norm_v1, norm_v2) / max(norm_v1, norm_v2)
            if ratio < ratio_min:
                continue

            # Rest-frame cytokinesis angle
            cos_theta = float(np.dot(v1_star, v2_star) / (norm_v1 * norm_v2))
            if cos_theta > cos_max:
                continue

            # Raw detector probability check
            p_m_d2 = 0.0
            if anchor_to_cand is not None:
                raw_m = int(anchor_to_cand[m])
                raw_d2 = int(anchor_to_cand[d2])
                p_m_d2 = prob_map.get((raw_m, raw_d2), 0.0)

            if p_m_d2 < p_cand_min:
                # Also check reverse mapping or relaxed probability
                # If no candidate edges exist at all, skip
                continue

            candidate_reassignments.append({
                "m": m, "d1": d1, "d2": d2, "c": c,
                "cos_theta": cos_theta,
                "sister_dist": sister_dist,
                "ratio": ratio,
                "p_m_d2": p_m_d2,
            })

    # Greedy disjoint selection: sort by cos_theta (most antiparallel first)
    candidate_reassignments.sort(key=lambda x: x["cos_theta"])

    assigned_mothers = set()
    assigned_daughters = set()
    reassign_edges_add = []
    reassign_edges_remove = []
    audit_items = []

    for cand in candidate_reassignments:
        m, d1, d2, c = cand["m"], cand["d1"], cand["d2"], cand["c"]
        # Disjoint: each mother and daughter used at most once
        if m in assigned_mothers or d2 in assigned_daughters:
            continue
        assigned_mothers.add(m)
        assigned_daughters.add(d2)

        reassign_edges_remove.append((c, d2))
        reassign_edges_add.append((m, d2))
        audit_items.append(cand)

    # Apply edits
    remove_set = set(reassign_edges_remove)
    new_edges_list = [e for e in edges if (int(e[0]), int(e[1])) not in remove_set]
    for u, v in reassign_edges_add:
        new_edges_list.append(np.array([u, v], dtype=edges.dtype))

    new_edges = np.array(new_edges_list, dtype=edges.dtype) if new_edges_list else edges[:0]

    audit = {
        "reassignments": len(audit_items),
        "candidates_scanned": candidates_scanned,
        "candidates_passing_geometry": len(candidate_reassignments),
        "items": audit_items,
    }
    return new_edges, audit


# ---------------------------------------------------------------------------
# build_candidate_graph: 032 smoothing + fork pruning + DC-CMR
# ---------------------------------------------------------------------------

def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group,
                          cand_npz=None):
    """Constructs verified Lineage candidate for a single dataset under 033 rules.

    Pipeline:
      1. Harmonized axial smoothing  (same as 032)
      2. Fork pruning DDKS-SPI       (same as 032)
      3. DC-CMR reassignment          (NEW in 033)
    """
    smoothed_coords, audit_smoothing = apply_harmonized_axial_regularizer(
        anchor.coords, anchor.edges, scale, variant_cfg.get("smoothing", {})
    )

    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors,
        variant_cfg.get("fork_pruning", {}), group=group
    )

    # NEW: DC-CMR post-processing
    reassign_cfg = variant_cfg.get("mitosis_reassignment", {})
    reassigned_edges, audit_reassign = apply_dc_cmr(
        smoothed_coords, pruned_edges, scale, reassign_cfg, cand_npz=cand_npz
    )

    cand = Lineage(smoothed_coords, reassigned_edges)
    cand.validate()
    return cand, {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
        "mitosis_reassignment": audit_reassign,
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


def _load_cand_npz(name, source_path):
    """Locate candidate NPZ for a dataset, checking multiple known paths."""
    search_paths = [
        source_path / "cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-005/cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent.parent / "A1688/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent.parent / "A1555/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1688/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1555/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
    ]
    for p in search_paths:
        if p.is_file():
            return np.load(p)
    return None


def _load_cand_npz_confirm(name, source_path):
    """Locate candidate NPZ for a confirmation dataset."""
    search_paths = [
        source_path / "cache-confirmation/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        source_path.parent.parent / "A1688/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        source_path.parent.parent / "A1555/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1688/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1555/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
    ]
    for p in search_paths:
        if p.is_file():
            return np.load(p)
    return None


def run_paired_evaluation(source_run, corpus, names, variant_cfg):
    """Runs strict label-free candidate generation, then official GT evaluation on remote node."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    source_path = Path(source_run)
    refs = {}
    cands = {}
    audits = {}
    metric = official_eval.official_metrics()

    for name in sorted(names):
        group = name.split("_", 1)[0]
        evidence_path = source_path / (name + ".evidence.npz")
        f = np.load(evidence_path)
        raw_coords = f["coords"]
        raw_edges = f["reference_edges"]
        desc = f["descriptors"] if "descriptors" in f else None
        stats = f["stats"] if "stats" in f else None
        scale = f["scale"]

        anchor = Lineage(raw_coords, raw_edges)
        anchor.validate()

        # Load candidate NPZ for DC-CMR probability lookup
        cand_npz = _load_cand_npz(name, source_path)

        # 1. Label-free candidate generation (smoothing + pruning + DC-CMR)
        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group,
            cand_npz=cand_npz
        )
        audits[name] = audit

        # 2. Open GT for official metric evaluation ONLY
        gt_path = Path(corpus["data_root"]) / (name + ".geff")
        if not gt_path.exists():
            local_alt = ROOT / "cache-gt" / (name + ".geff")
            if local_alt.exists():
                gt_path = local_alt
            else:
                raise FileNotFoundError(f"GT geff missing: {gt_path}")

        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            estimated = float(len(anchor.coords))

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
        spi_cnt = audit["fork_pruning"].get("spi_triggered", 0)
        ra_cnt = audit["mitosis_reassignment"].get("reassignments", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt} "
              f"(spi={spi_cnt}), reassign={ra_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-033 on A1555."""
    cfg = config()
    start_time = time.time()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    variants = cfg["variants"]

    print(f"Starting Campaign 033 (DC-CMR) on {args.node} across {len(names)} development datasets...")
    results = {}

    for v_key, v_cfg in variants.items():
        print(f"\n--- Evaluating {v_key}: {v_cfg['description']} ---")
        v_res = run_paired_evaluation(source, corpus, names, v_cfg)
        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        ra_total = sum(v_res["audits"][n]["mitosis_reassignment"]["reassignments"] for n in names)
        print(f"Variant {v_key} Result: score={cand_score:.6f} (ref={ref_score:.6f}, "
              f"delta={delta:+.6f}), reassignments={ra_total}, gate_passed={passed}")
        results[v_key] = v_res

    elapsed = time.time() - start_time

    # Determine winning variant
    passed_variants = [k for k, r in results.items() if r["gate"]["passed"]]
    if passed_variants:
        best_v = max(passed_variants, key=lambda k: results[k]["evaluation"]["summary"]["candidate"]["score"])
        disposition = "eligible_for_confirmation_and_package"
    else:
        best_v = max(results.keys(), key=lambda k: results[k]["evaluation"]["summary"]["candidate"]["score"])
        disposition = "rejected_no_kaggle_submission"

    print(f"\nCampaign 033 Evaluation Summary:")
    print(f"Best Variant: {best_v} (disposition: {disposition})")

    summary = {
        "disposition": disposition,
        "best_variant": best_v,
        "variants_evaluated": list(variants.keys()),
        "paired_evaluations": {k: results[k]["evaluation"] for k in results},
        "gates": {k: results[k]["gate"] for k in results},
        "audits": {k: results[k]["audits"] for k in results},
        "seconds": elapsed,
        "source_run": cfg["source_run"],
        "code_hash": code_hash(),
        "safety": {"label_free_candidate_generation": True, "no_gt_leak": True}
    }
    dump(Path(run) / "temporal_association_evaluation.json", summary)
    print(f"Campaign 033 completed in {elapsed:.1f}s.")
    return summary


def run_confirmation_evaluation(source_run, plan, names, variant_cfg):
    """Runs strict label-free candidate generation on confirmation fold, then official evaluation."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    source_path = Path(source_run)
    ref_csv = source_path / "cache-confirmation/reference.csv"
    references = load_reference(ref_csv, names)
    scale = np.array([1.625, 0.40625, 0.40625], dtype=float)

    refs = {}
    cands = {}
    audits = {}
    metric = official_eval.official_metrics()

    for name in sorted(names):
        group = name.split("_", 1)[0]
        anchor = references[name]

        # Load candidate NPZ for DC-CMR
        cand_npz = _load_cand_npz_confirm(name, source_path)

        # 1. Label-free candidate generation (smoothing + pruning + DC-CMR)
        cand_graph, audit = build_candidate_graph(
            anchor, scale, None, {}, variant_cfg, group=group,
            cand_npz=cand_npz
        )
        audits[name] = audit

        # 2. Open GT for official metric evaluation ONLY
        gt_path = Path(plan["data_root"]) / (name + ".geff")
        if not gt_path.exists():
            local_alt = source_path / "cache-confirmation/raw" / (name + ".geff")
            if local_alt.exists():
                gt_path = local_alt
            else:
                raise FileNotFoundError(f"GT geff missing: {gt_path}")

        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            estimated = float(len(anchor.coords))

        row = {"name": name, "group": group, "split": "confirmation",
               "scale_um": scale.tolist(), "gt": str(gt_path)}

        def eval_graph(g):
            nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(g.coords)}
            edges_list = [{"source_id": int(u), "target_id": int(v)} for u, v in g.edges]
            pred = make_graph(*as_evaluator_inputs(nodes, edges_list, round_coordinates=False))
            counts = metric.evaluate(pred, gt, scale=tuple(scale))
            values = metric.per_sample_metrics(counts, estimated, metric.node_recall(pred, gt))
            return {**values, "dataset": name, "group": row["group"],
                    "split": "confirmation", "counts": counts._asdict()}

        refs[name] = eval_graph(anchor)
        cands[name] = eval_graph(cand_graph)
        ref_adj = refs[name]["adj_edge_jaccard"]
        cand_adj = cands[name]["adj_edge_jaccard"]
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        pr_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        spi_cnt = audit["fork_pruning"].get("spi_triggered", 0)
        ra_cnt = audit["mitosis_reassignment"].get("reassignments", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt} "
              f"(spi={spi_cnt}), reassign={ra_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "confirmation")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_confirmation_job(args, run, source):
    """Main execution entrypoint for confirm-temporal-033 on A1555."""
    cfg = config()
    start_time = time.time()
    plan = verify_sources(source, code_manifest())
    names = plan["panels"]["confirmation"]

    print(f"Starting Campaign 033 Confirmation Gate on {args.node} across {len(names)} confirmation datasets...")
    evaluations = {}
    best_variant = None
    best_delta = -float("inf")
    best_res = None

    for v_name, v_cfg in cfg["variants"].items():
        print(f"\n--- Evaluating Confirmation Variant {v_name} ---")
        v_res = run_confirmation_evaluation(source, plan, names, v_cfg)
        evaluations[v_name] = v_res

        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        cand_div_tp = v_res["evaluation"]["summary"]["candidate"]["division_tp"]
        ref_div_tp = v_res["evaluation"]["summary"]["reference"]["division_tp"]
        cand_div_fp = v_res["evaluation"]["summary"]["candidate"]["division_fp"]
        ref_div_fp = v_res["evaluation"]["summary"]["reference"]["division_fp"]
        ra_total = sum(v_res["audits"][n]["mitosis_reassignment"]["reassignments"] for n in names)
        print(f"Confirmation {v_name}: score={cand_score:.6f} (ref={ref_score:.6f}, delta={delta:+.6f}), "
              f"div_tp={cand_div_tp}/{ref_div_tp}, div_fp={cand_div_fp}/{ref_div_fp}, "
              f"reassign={ra_total}, gate_passed={passed}")

        if passed and (best_res is None or not best_res["gate"]["passed"] or delta > best_delta):
            best_delta = delta
            best_variant = v_name
            best_res = v_res
        elif best_res is None or (not best_res["gate"]["passed"] and delta > best_delta):
            best_delta = delta
            best_variant = v_name
            best_res = v_res

    target_variant = cfg.get("target_variant", best_variant)
    primary_res = evaluations[target_variant]
    champion_passed = primary_res["gate"]["passed"]
    elapsed = time.time() - start_time
    disposition = "eligible_for_final_package_and_submission" if champion_passed else "rejected_no_kaggle_submission"

    summary = {
        "disposition": disposition,
        "champion_variant": target_variant,
        "recipe": cfg["variants"][target_variant]["description"],
        "result": primary_res,
        "all_variants": {k: {"score": v["evaluation"]["summary"]["candidate"]["score"],
                             "delta": v["evaluation"]["summary"]["candidate"]["score"] - v["evaluation"]["summary"]["reference"]["score"],
                             "gate_passed": v["gate"]["passed"],
                             "div_tp": v["evaluation"]["summary"]["candidate"]["division_tp"],
                             "div_fp": v["evaluation"]["summary"]["candidate"]["division_fp"]}
                         for k, v in evaluations.items()},
        "seconds": elapsed,
        "source_run": cfg["source_run"],
        "code_hash": code_hash(),
        "safety": {"label_free_candidate_generation": True, "no_gt_leak": True}
    }
    dump(Path(run) / "confirmation_evaluation.json", summary)
    print(f"\nCampaign 033 Confirmation completed in {elapsed:.1f}s.")
    print(f"Target Variant: {target_variant}, Delta: {summary['all_variants'][target_variant]['delta']:+.6f}, Disposition: {disposition}")
    return summary
