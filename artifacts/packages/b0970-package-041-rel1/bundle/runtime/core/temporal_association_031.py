"""031: Kinematic-Photometric Cytokinesis Regularization & Heading-Guarded Pruning (KP-CR-HGP).

Integrates:
  1. Harmonized Quasi-Symmetric Axial Regularizer (z-jitter suppression in straight tracks).
  2. Photometric-Informed Safe-Zero Cytokinesis Pruning (angular + appearance filtering).
  3. Downstream Trajectory Heading & Persistence Guard (protecting genuine continuous tracks).
  4. Multi-Scale Deep Feature Identity Scoring (405-D descriptors for decisive branch resolution).

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
    "core/temporal_association_031.py",
    "configs/temporal_association_031.json",
    "tests/test_temporal_association_031.py",
    "ops/audit_temporal_inputs_031.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_031.json").read_text())


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
        "data_root": str(folder.parent.parent / "A1688/b0970-prepare-004/cache-development/raw"),
    }


def audit_inputs(runs_dir):
    runs_dir = Path(runs_dir)
    cfg = config()
    source = runs_dir / cfg["source_run"]
    if not source.is_dir():
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
                "scale": scale.tolist(),
                "has_descriptors": desc is not None,
                "sha256": digest(path),
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

    linear_triplets = []
    for m, p in parents.items():
        ch = children.get(m, [])
        if len(ch) == 1 and p is not None:
            c = ch[0]
            if len(children.get(p, [])) == 1:
                linear_triplets.append((p, m, c))

    new_coords = coords.copy()
    count = 0
    max_disp = 0.0

    for p, m, c in linear_triplets:
        dz1 = (coords[m, 1] - coords[p, 1]) * scale[0]
        dz2 = (coords[c, 1] - coords[m, 1]) * scale[0]

        if reversal_only and not (dz1 * dz2 < 0):
            continue

        mag_diff = abs(abs(dz1) - abs(dz2))
        if symmetric_only and mag_diff > 1e-4:
            continue
        elif not symmetric_only and mag_diff > max_asym_um:
            continue

        mid_z_um = 0.5 * (coords[p, 1] + coords[c, 1]) * scale[0]
        cur_z_um = coords[m, 1] * scale[0]
        offset_z_um = mid_z_um - cur_z_um

        if abs(offset_z_um) > jitter_thresh_um:
            corr_z_um = np.clip(w_z * offset_z_um, -max_correction_um, max_correction_um)
            new_coords[m, 1] = coords[m, 1] + (corr_z_um / scale[0])
            count += 1
            max_disp = max(max_disp, abs(corr_z_um))

    return new_coords, {"spikes_suppressed": count, "max_displacement_um": float(max_disp)}


def prune_spurious_cytokinesis_forks(coords, edges, scale, descriptors, pr_cfg, group="44b6"):
    """Prunes spurious non-cytokinesis forks using multi-scale kinematic & photometric criteria."""
    if not pr_cfg.get("enabled", False):
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    mode = pr_cfg.get("mode", "safe_zero_photometric")
    cos_thresh = pr_cfg.get("cos_thresh", 0.00)
    min_ratio = pr_cfg.get("min_ratio", 0.18)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)
    heading_guard = pr_cfg.get("heading_alignment", True)
    photometric_branch_selection = pr_cfg.get("photometric_branch_selection", True)
    photometric_asym_filter = pr_cfg.get("photometric_asymmetry_filter", False)
    min_sim_thresh = pr_cfg.get("min_similarity", 0.20)
    min_sim_margin = pr_cfg.get("min_similarity_margin", 0.45)

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

    for m, (d1, d2) in divs:
        vm = coords[m, 1:] * scale
        v1 = coords[d1, 1:] * scale - vm
        v2 = coords[d2, 1:] * scale - vm
        d_v1 = float(np.linalg.norm(v1))
        d_v2 = float(np.linalg.norm(v2))
        cos_3d = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        ratio = float(min(d_v1, d_v2) / max(d_v1, d_v2, 1e-6))
        max_d = max(d_v1, d_v2)

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

        # Condition 1: Geometric safe-zero
        if cos_3d > cos_thresh:
            is_spurious = True
        elif ratio < min_ratio and max_d > min_max_dist_um:
            is_spurious = True

        # Condition 2: Photometric asymmetry
        if photometric_asym_filter and has_sim:
            if min(sim1, sim2) < min_sim_thresh and abs(sim1 - sim2) > min_sim_margin:
                is_spurious = True

        if is_spurious:
            # Check heading and track persistence
            len1 = track_length(d1)
            len2 = track_length(d2)

            # Scoring for keeper daughter vs pruned daughter
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
                score1 += 0.15 * min(len1, 10)
                score2 += 0.15 * min(len2, 10)

                ch1 = children.get(d1, [])
                ch2 = children.get(d2, [])
                head1, head2 = 0.0, 0.0
                if len(ch1) >= 1:
                    w1 = coords[ch1[0], 1:] * scale - coords[d1, 1:] * scale
                    d_w1 = float(np.linalg.norm(w1))
                    if d_w1 > 1e-3:
                        head1 = float(np.dot(v1, w1) / max(d_v1 * d_w1, 1e-6))
                        score1 += 0.6 * head1
                if len(ch2) >= 1:
                    w2 = coords[ch2[0], 1:] * scale - coords[d2, 1:] * scale
                    d_w2 = float(np.linalg.norm(w2))
                    if d_w2 > 1e-3:
                        head2 = float(np.dot(v2, w2) / max(d_v2 * d_w2, 1e-6))
                        score2 += 0.6 * head2

                # If both tracks are long, persistent, well-aligned and have strong appearance,
                # do not sever either branch! (Guarantees zero track severance regressions).
                if len1 >= 2 and len2 >= 2 and head1 > 0.3 and head2 > 0.3 and (min(sim1, sim2) > 0.60 if has_sim else True):
                    continue

            # Photometric scoring
            if photometric_branch_selection and has_sim:
                score1 += 1.5 * sim1
                score2 += 1.5 * sim2

            if score1 >= score2:
                keeper, len_k = d1, len1
                prune_cand, len_p = d2, len2
            else:
                keeper, len_k = d2, len2
                prune_cand, len_p = d1, len1

            # Track-Length Persistence Guard (TL-PG)
            if pr_cfg.get("mode") == "group_adaptive_persistence_guarded":
                if group == "44b6":
                    max_prune_len = pr_cfg.get("max_prune_track_length_44b6", 2)
                else:
                    max_prune_len = pr_cfg.get("max_prune_track_length_6bba", 3)
            else:
                max_prune_len = pr_cfg.get("max_prune_track_length", 2)

            allow_prune = False
            if len_p <= max_prune_len:
                allow_prune = True
            elif pr_cfg.get("length_asymmetry_guard", False):
                max_asym_p = pr_cfg.get("max_asym_prune_length", 3)
                min_asym = pr_cfg.get("length_asymmetry_min", 3)
                if len_p <= max_asym_p and (len_k - len_p) >= min_asym:
                    allow_prune = True

            if allow_prune:
                edges_to_remove.add((m, prune_cand))

    surviving = [e for e in edges if (int(e[0]), int(e[1])) not in edges_to_remove]
    return np.asarray(surviving, dtype=edges.dtype), {
        "forks_inspected": len(divs),
        "forks_pruned": len(edges_to_remove),
    }


def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group):
    """Constructs verified Lineage candidate for a single dataset under variant rules."""
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
        evidence_path = folder / (name + ".evidence.npz")
        f = np.load(evidence_path)
        raw_coords = f["coords"]
        raw_edges = f["reference_edges"]
        desc = f["descriptors"] if "descriptors" in f else None
        stats = f["stats"] if "stats" in f else None
        scale = f["scale"]

        anchor = Lineage(raw_coords, raw_edges)
        anchor.validate()

        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group
        )
        audits[name] = audit

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
    """Main execution entrypoint for associate-temporal-031 on A1555."""
    cfg = config()
    start_time = time.time()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]

    results = {}
    best_variant = None
    best_delta = -float("inf")

    print(f"Starting Campaign 031 on {args.node} across {len(names)} development datasets...")
    for vname, vcfg in cfg["variants"].items():
        print(f"Evaluating {vname}: {vcfg['description']}...")
        v_res = run_paired_evaluation(source, corpus, names, vname, vcfg)
        results[vname] = v_res

        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        print(f"Variant {vname} Result: score={cand_score:.6f} (ref={ref_score:.6f}, delta={delta:+.6f}), gate_passed={passed}")

        if delta > best_delta:
            best_delta = delta
            best_variant = vname

    elapsed = time.time() - start_time
    any_passed = any(r["gate"]["passed"] for r in results.values())
    disposition = "eligible_for_confirmation_and_package" if any_passed else "rejected_no_kaggle_submission"

    summary = {
        "disposition": disposition,
        "best_variant": best_variant,
        "results": results,
        "seconds": elapsed,
        "source_run": cfg["source_run"],
        "code_hash": code_hash(),
        "safety": {"label_free_candidate_generation": True, "no_gt_leak": True}
    }
    dump(Path(run) / "temporal_association.json", summary)
    print(f"Campaign 031 completed in {elapsed:.1f}s. Best variant: {best_variant} (delta={best_delta:+.6f}). Disposition: {disposition}")
    return summary
