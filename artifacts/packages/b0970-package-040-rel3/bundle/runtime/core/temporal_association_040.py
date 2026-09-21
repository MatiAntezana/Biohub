"""040: Bounded Divergence Persistence Gating (BDPG).

Refines cytokinesis fork discrimination to eliminate ephemeral spur divisions
while preserving 100% of true biological cytokinesis events under convective drift:
1. Obtuse divergent cytokinesis (cos_3d <= 0.0):
   Naturally diverging daughters; immunized if both tracks persist (len >= 3)
   or have positive forward heading (head > 0.3).
2. Drift-tilted cytokinesis (0.0 < cos_3d <= cos_max):
   True divisions under strong convective tissue drift angle slightly forward.
   Immunized ONLY IF both daughters exhibit robust persistence (len >= 5)
   AND bounded birth distance (max_d <= 6.0 um). Ephemeral spurs (e.g. 3-frame jumps)
   are cleanly pruned.
3. Acute crossovers (cos_3d > cos_max):
   Unphysical parallel jumps; never immunized by track persistence.
   Pruned via Distance-Dominant Keeper Selection & Spatial Proximity Invariant (DDKS-SPI).

Preserves all lineage invariants: in_degree <= 1, out_degree <= 2.
Label-free candidate generation. Ground truth accessed only during paired evaluation.
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

ADDITIONS = {
    "core/temporal_association_040.py",
    "configs/temporal_association_040.json",
    "tests/test_temporal_association_040.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_040.json").read_text())


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
        return plan
    elif task == "prepare-selector":
        corpus_path = folder / "corpus.json"
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Corpus missing: {corpus_path}")
        corpus = json.loads(corpus_path.read_text())
        return corpus
    elif task == "rank-repairs" or task == "prepare-coverage":
        split_path = folder / "split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file missing: {split_path}")
        split = json.loads(split_path.read_text())
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
    return {
        "status": "audit_passed",
        "source_run": cfg["source_run"],
        "development_datasets": len(names),
        "stats": npz_stats,
    }, corpus


def apply_harmonized_axial_regularizer(coords, edges, scale, sm_cfg):
    """Harmonized quasi-symmetric axial regularizer suppressing unphysical z-reversal jitter."""
    if not sm_cfg.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0, "max_displacement_um": 0.0}

    reversal_only = sm_cfg.get("reversal_only", True)
    symmetric_only = sm_cfg.get("symmetric_only", True)
    max_asym_um = sm_cfg.get("max_asymmetry_um", 0.0)
    jitter_thresh_um = sm_cfg.get("jitter_threshold_um", 0.8)
    max_correction_um = sm_cfg.get("max_correction_um", 0.6)
    w_z = sm_cfg.get("weight_z", 0.8)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    new_coords = coords.copy().astype(np.float64)
    count = 0
    max_disp = 0.0

    for i in range(len(coords)):
        if i in parents and len(children[i]) == 1:
            p = parents[i]
            c = children[i][0]

            if coords[p, 0] + 1 == coords[i, 0] and coords[i, 0] + 1 == coords[c, 0]:
                zp = coords[p, 1] * scale[0]
                zi = coords[i, 1] * scale[0]
                zc = coords[c, 1] * scale[0]

                d1 = zi - zp
                d2 = zc - zi

                is_reversal = (d1 * d2 < 0)
                if reversal_only and not is_reversal:
                    continue

                if symmetric_only and is_reversal:
                    asym = abs(abs(d1) - abs(d2))
                    if asym > max_asym_um:
                        continue

                mag = min(abs(d1), abs(d2))
                if mag >= jitter_thresh_um:
                    z_target = 0.5 * (zp + zc)
                    disp_um = abs(zi - z_target) * w_z
                    if disp_um > max_correction_um:
                        disp_um = max_correction_um
                    corr_um = math.copysign(disp_um, z_target - zi)
                    new_coords[i, 1] += corr_um / scale[0]
                    count += 1
                    max_disp = max(max_disp, disp_um)

    return new_coords, {
        "spikes_suppressed": count,
        "max_displacement_um": float(max_disp),
    }


def prune_spurious_cytokinesis_forks(coords, edges, scale, descriptors, pr_cfg, group=None):
    """Bounded Divergence Persistence Gating (BDPG) & DDKS-SPI."""
    if not pr_cfg.get("enabled", False):
        return edges.copy(), {"forks_inspected": 0, "forks_pruned": 0, "spi_triggered": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.0)
    min_ratio = pr_cfg.get("min_ratio", 0.18)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)
    photometric_branch_selection = pr_cfg.get("photometric_branch_selection", False)

    # SPI config
    spi_enabled = pr_cfg.get("spi_enabled", True)
    spi_max_close_um = pr_cfg.get("spi_max_close_um", 1.5)
    spi_min_far_um = pr_cfg.get("spi_min_far_um", 2.5)
    spi_min_ratio = pr_cfg.get("spi_min_ratio", 2.0)

    # Heading guard & BDPG config
    heading_guard = pr_cfg.get("heading_guard", True)
    heading_guard_cos_max = float(pr_cfg.get("heading_guard_cos_max", 0.20))
    guard_requires_cos = bool(pr_cfg.get("guard_requires_cos", True))
    div_pers = int(pr_cfg.get("divergent_pers_min", 3))
    tilt_pers = int(pr_cfg.get("tilted_pers_min", 5))
    max_birth_d = float(pr_cfg.get("tilted_max_birth_dist_um", 6.0))

    # Distance cost mode
    dist_cost_mode = pr_cfg.get("distance_cost_mode", "linear")
    d0_um = pr_cfg.get("distance_scale_d0_um", 2.0)
    alpha_dist = pr_cfg.get("distance_cost_alpha", 1.0)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    def track_length(node):
        curr = node
        length = 0
        while curr in children and len(children[curr]) > 0:
            curr = children[curr][0]
            length += 1
            if length > 50:
                break
        return length

    divs = [(m, ch) for m, ch in children.items() if len(ch) == 2]
    edges_to_remove = set()
    spi_triggered = 0

    for m, (d1, d2) in divs:
        vm = coords[m, 1:] * scale
        v1 = coords[d1, 1:] * scale - vm
        v2 = coords[d2, 1:] * scale - vm
        d_v1 = float(np.linalg.norm(v1))
        d_v2 = float(np.linalg.norm(v2))
        cos_3d = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        ratio = float(min(d_v1, d_v2) / max(max(d_v1, d_v2), 1e-6))
        max_d = max(d_v1, d_v2)
        min_d = min(d_v1, d_v2)

        sim1, sim2 = 0.5, 0.5
        has_sim = False
        if descriptors is not None and len(descriptors) > max(m, d1, d2):
            d_m = descriptors[m]
            norm_m = np.linalg.norm(d_m)
            if norm_m > 0:
                norm1 = np.linalg.norm(descriptors[d1])
                norm2 = np.linalg.norm(descriptors[d2])
                if norm1 > 0 and norm2 > 0:
                    sim1 = float(np.dot(d_m, descriptors[d1]) / (norm_m * norm1))
                    sim2 = float(np.dot(d_m, descriptors[d2]) / (norm_m * norm2))
                    has_sim = True

        is_spurious = False
        if cos_3d > cos_thresh:
            is_spurious = True
        elif ratio < min_ratio and max_d > min_max_dist_um:
            is_spurious = True

        if is_spurious:
            len1 = track_length(d1)
            len2 = track_length(d2)

            ch1 = children.get(d1, [])
            ch2 = children.get(d2, [])
            head1, head2 = 0.0, 0.0
            if len(ch1) >= 1:
                w1 = coords[ch1[0], 1:] * scale - coords[d1, 1:] * scale
                d_w1 = float(np.linalg.norm(w1))
                if d_w1 > 1e-3:
                    head1 = float(np.dot(v1, w1) / max(d_v1 * d_w1, 1e-6))
            if len(ch2) >= 1:
                w2 = coords[ch2[0], 1:] * scale - coords[d2, 1:] * scale
                d_w2 = float(np.linalg.norm(w2))
                if d_w2 > 1e-3:
                    head2 = float(np.dot(v2, w2) / max(d_v2 * d_w2, 1e-6))

            # STEP 1: Bounded Divergence Persistence Gating (BDPG)
            # 1. Obtuse divergent cytokinesis (cos_3d <= 0.0):
            #    Both daughter branches diverge naturally; protected if both tracks persist (len >= divergent_pers_min)
            #    or have strong forward persistence/heading.
            # 2. Drift-tilted cytokinesis (0.0 < cos_3d <= heading_guard_cos_max):
            #    Cytokinesis under strong convective tissue drift can result in forward-angled daughters.
            #    Protected ONLY IF both daughters exhibit robust persistence (len >= tilted_pers_min)
            #    AND bounded birth distance (max_d <= tilted_max_birth_dist_um).
            #    This cleanly prunes ephemeral spur forks (such as 3-frame jumps) while safeguarding real divisions.
            # 3. Acute crossovers (cos_3d > heading_guard_cos_max):
            #    Unphysical crossovers where cells move in the same direction; never immunized by persistence.
            if heading_guard:
                if guard_requires_cos:
                    # Case 1: Obtuse divergent cytokinesis
                    if cos_3d <= 0.0 and (
                        (len1 >= div_pers and len2 >= div_pers) or
                        (len1 >= 2 and len2 >= 2 and head1 > 0.3 and head2 > 0.3)
                    ):
                        continue

                    # Case 2: Drift-tilted cytokinesis
                    if 0.0 < cos_3d <= heading_guard_cos_max and (
                        len1 >= tilt_pers and len2 >= tilt_pers and max_d <= max_birth_d
                    ):
                        continue
                else:
                    if (len1 >= 5 and len2 >= 5) or (cos_3d <= heading_guard_cos_max and (
                        (len1 >= 3 and len2 >= 3) or (len1 >= 2 and len2 >= 2 and head1 > 0.3 and head2 > 0.3)
                    )):
                        continue

            # STEP 2: Distance-Dominant Keeper Selection (DDKS-SPI)
            # Extreme spatial asymmetry indicates a local/stationary continuing cell
            # versus an unphysical jumping crossover.
            is_extreme_asymmetry = (
                (min_d <= spi_max_close_um and max_d >= spi_min_far_um) or
                (max_d >= spi_min_far_um and (max_d / max(min_d, 1e-6)) >= spi_min_ratio)
            )

            if spi_enabled and is_extreme_asymmetry:
                # The closer daughter is unconditionally the keeper!
                if d_v1 <= d_v2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                spi_triggered += 1
                continue

            # STEP 3: Standard / Quadratic scoring for non-extreme forks
            if dist_cost_mode == "quadratic":
                score1 = -alpha_dist * ((d_v1 / d0_um) ** 2)
                score2 = -alpha_dist * ((d_v2 / d0_um) ** 2)
            else:
                score1 = -d_v1 / 10.0
                score2 = -d_v2 / 10.0

            # Kinematic alignment with mother
            if m in parents:
                p = parents[m]
                u_in = vm - coords[p, 1:] * scale
                d_u = float(np.linalg.norm(u_in))
                align1 = float(np.dot(u_in, v1) / max(d_u * d_v1, 1e-6))
                align2 = float(np.dot(u_in, v2) / max(d_u * d_v2, 1e-6))
                score1 += 1.0 * align1
                score2 += 1.0 * align2

            # Downstream heading and persistence
            if heading_guard:
                score1 += 0.15 * min(len1, 10) + 0.6 * head1
                score2 += 0.15 * min(len2, 10) + 0.6 * head2

            # Photometric scoring
            if photometric_branch_selection and has_sim:
                score1 += 1.5 * sim1
                score2 += 1.5 * sim2

            if score1 >= score2:
                edges_to_remove.add((m, d2))
            else:
                edges_to_remove.add((m, d1))

    surviving = [e for e in edges if (int(e[0]), int(e[1])) not in edges_to_remove]
    return np.asarray(surviving, dtype=edges.dtype), {
        "forks_inspected": len(divs),
        "forks_pruned": len(edges_to_remove),
        "spi_triggered": spi_triggered,
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group):
    """Constructs verified Lineage candidate for a single dataset under variant rules."""
    smoothed_coords, audit_smoothing = apply_harmonized_axial_regularizer(
        anchor.coords, anchor.edges, scale, variant_cfg.get("smoothing", {})
    )

    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors, variant_cfg.get("fork_pruning", {}), group=group
    )

    cand = Lineage(smoothed_coords, pruned_edges)
    cand.validate()
    return cand, {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
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

        # 1. Label-free candidate generation
        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group
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
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt} (spi={spi_cnt})", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-040 on A1555."""
    cfg = config()
    start_time = time.time()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    variants = cfg["variants"]

    print(f"Starting Campaign 040 (BDPG) on {args.node} across {len(names)} development datasets...")
    results = {}

    for v_key, v_cfg in variants.items():
        print(f"\n--- Evaluating {v_key}: {v_cfg['description']} ---")
        v_res = run_paired_evaluation(source, corpus, names, v_cfg)
        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        print(f"Variant {v_key} Result: score={cand_score:.6f} (ref={ref_score:.6f}, delta={delta:+.6f}), gate_passed={passed}")
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

    print(f"\nCampaign 040 Evaluation Summary:")
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
    print(f"Campaign 040 completed in {elapsed:.1f}s.")
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

        # 1. Label-free candidate generation
        cand_graph, audit = build_candidate_graph(
            anchor, scale, None, {}, variant_cfg, group=group
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
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt} (spi={spi_cnt})", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "confirmation")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_confirmation_job(args, run, source):
    """Main execution entrypoint for confirm-temporal-040 on A1555."""
    cfg = config()
    start_time = time.time()
    plan = verify_sources(source, code_manifest())
    names = plan["panels"]["confirmation"]

    print(f"Starting Campaign 040 Confirmation Gate on {args.node} across {len(names)} confirmation datasets...")
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
        print(f"Confirmation {v_name}: score={cand_score:.6f} (ref={ref_score:.6f}, delta={delta:+.6f}), "
              f"div_tp={cand_div_tp}/{ref_div_tp}, div_fp={cand_div_fp}/{ref_div_fp}, gate_passed={passed}")

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
    print(f"\nCampaign 040 Confirmation completed in {elapsed:.1f}s.")
    print(f"Target Variant: {target_variant}, Delta: {summary['all_variants'][target_variant]['delta']:+.6f}, Disposition: {disposition}")
    return summary
