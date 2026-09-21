"""025: Momentum-Conserving Crossover Untangling and Biomechanical Temporal Association.

Combines:
  1. Pure Symmetric Axial Regularizer (z-jitter suppression in isolated straight trajectories).
  2. Momentum-Conserving Crossover Untangling (resolving track swaps at cell-cell spatial intersections).
  3. Spurious Mitosis Pruning (anti-cytokinesis divergence and extreme distance asymmetry).

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
    "core/temporal_association_025.py",
    "configs/temporal_association_025.json",
    "tests/test_temporal_association_025.py",
    "ops/audit_temporal_inputs_025.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_025.json").read_text())


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

    symmetric_only = cfg_smoothing.get("symmetric_only", True)
    jitter_thresh_um = cfg_smoothing.get("jitter_threshold_um", 0.8)
    max_corr_um = cfg_smoothing.get("max_correction_um", 0.6)
    w_z = cfg_smoothing.get("weight_z", 0.80)

    in_edges = defaultdict(list)
    out_edges = defaultdict(list)
    for u, v in edges:
        in_edges[int(v)].append(int(u))
        out_edges[int(u)].append(int(v))

    smoothed = coords.copy()
    linear_nodes = [
        i for i in range(len(coords))
        if len(in_edges[i]) == 1 and len(out_edges[i]) == 1
    ]

    spikes_suppressed = 0
    for nid in linear_nodes:
        p = in_edges[nid][0]
        s = out_edges[nid][0]
        if coords[nid, 0] != coords[p, 0] + 1 or coords[s, 0] != coords[nid, 0] + 1:
            continue

        zp = coords[p, 1] * scale[0]
        zc = coords[nid, 1] * scale[0]
        zs = coords[s, 1] * scale[0]

        dz_in = zc - zp
        dz_out = zs - zc

        if dz_in * dz_out < 0:
            if symmetric_only:
                asym = abs(abs(dz_in) - abs(dz_out)) / max(abs(dz_in) + abs(dz_out), 1e-6)
                if asym > 0.35:
                    continue

            jitter_mag = min(abs(dz_in), abs(dz_out))
            if jitter_mag >= jitter_thresh_um:
                z_mid = 0.5 * (zp + zs)
                correction = (z_mid - zc) * w_z
                clamped_corr = np.clip(correction, -max_corr_um, max_corr_um)
                smoothed[nid, 1] = (zc + clamped_corr) / scale[0]
                spikes_suppressed += 1

    return smoothed, {"spikes_suppressed": spikes_suppressed}


def untangle_crossover_swaps(coords, edges, scale, descriptors, xo_cfg):
    """Untangles crossing trajectories by enforcing momentum conservation and trajectory straightness."""
    if not xo_cfg.get("enabled", False):
        return edges, {"crossovers_inspected": 0, "crossovers_untangled": 0}

    max_spatial_dist_um = xo_cfg.get("max_spatial_dist_um", 5.5)
    min_delta_momentum = xo_cfg.get("min_delta_momentum", 1.20)
    max_cur_cos = xo_cfg.get("max_cur_cos", 0.10)
    min_swap_cos = xo_cfg.get("min_swap_cos", 0.60)
    max_swaps = xo_cfg.get("max_swaps_per_dataset", 50)
    use_desc = xo_cfg.get("use_descriptors", False)
    min_desc_delta = xo_cfg.get("min_desc_sim_delta", -0.10)

    in_edges = defaultdict(list)
    out_edges = defaultdict(list)
    for u, v in edges:
        in_edges[int(u)]
        out_edges[int(u)].append(int(v))
        in_edges[int(v)].append(int(u))

    parents = {int(v): int(u[0]) for v, u in in_edges.items() if len(u) == 1}
    children = {int(u): int(v[0]) for u, v in out_edges.items() if len(v) == 1}

    # Group linear edges by frame t: (u, v) where u has parent and v has child
    by_t = defaultdict(list)
    for u, v in edges:
        u_int, v_int = int(u), int(v)
        if u_int in children and len(out_edges[u_int]) == 1 and len(in_edges[v_int]) == 1:
            if u_int in parents and v_int in children:
                t = int(coords[u_int, 0])
                by_t[t].append((u_int, v_int))

    swapped_u = set()
    swapped_v = set()
    swaps_performed = []
    crossovers_inspected = 0

    for t in sorted(by_t.keys()):
        edge_pairs = by_t[t]
        if len(edge_pairs) < 2:
            continue

        for i in range(len(edge_pairs)):
            if len(swaps_performed) >= max_swaps:
                break
            u1, v1 = edge_pairs[i]
            if u1 in swapped_u or v1 in swapped_v:
                continue

            c_u1 = coords[u1, 1:] * scale
            c_v1 = coords[v1, 1:] * scale
            p1 = parents[u1]
            c_p1 = coords[p1, 1:] * scale

            v_in1 = c_u1 - c_p1
            v_out1 = c_v1 - c_u1
            d_in1 = np.linalg.norm(v_in1)
            d_out1 = np.linalg.norm(v_out1)
            if d_in1 < 0.2 or d_out1 < 0.2:
                continue
            cos1 = float(np.dot(v_in1, v_out1) / (d_in1 * d_out1))

            for j in range(i + 1, len(edge_pairs)):
                if len(swaps_performed) >= max_swaps:
                    break
                u2, v2 = edge_pairs[j]
                if u2 in swapped_u or v2 in swapped_v:
                    continue

                c_u2 = coords[u2, 1:] * scale
                # Spatial proximity check
                d_u1_u2 = float(np.linalg.norm(c_u1 - c_u2))
                if d_u1_u2 > max_spatial_dist_um:
                    continue

                c_v2 = coords[v2, 1:] * scale
                p2 = parents[u2]
                c_p2 = coords[p2, 1:] * scale

                v_in2 = c_u2 - c_p2
                v_out2 = c_v2 - c_u2
                d_in2 = np.linalg.norm(v_in2)
                d_out2 = np.linalg.norm(v_out2)
                if d_in2 < 0.2 or d_out2 < 0.2:
                    continue
                cos2 = float(np.dot(v_in2, v_out2) / (d_in2 * d_out2))

                crossovers_inspected += 1

                # Current assignment must be deficient (hairpin or sharp bend)
                if max(cos1, cos2) > max_cur_cos and (cos1 + cos2) > 0.0:
                    continue

                # Swapped velocities: u1 -> v2 and u2 -> v1
                v_swap1 = c_v2 - c_u1
                v_swap2 = c_v1 - c_u2
                d_s1 = np.linalg.norm(v_swap1)
                d_s2 = np.linalg.norm(v_swap2)
                if d_s1 > 8.0 or d_s2 > 8.0:
                    continue

                cos_s1 = float(np.dot(v_in1, v_swap1) / (d_in1 * d_s1))
                cos_s2 = float(np.dot(v_in2, v_swap2) / (d_in2 * d_s2))

                # Swapped assignment must be strongly directional
                if min(cos_s1, cos_s2) < min_swap_cos:
                    continue

                cur_momentum = cos1 + cos2
                swap_momentum = cos_s1 + cos_s2
                delta_m = swap_momentum - cur_momentum

                if delta_m < min_delta_momentum:
                    continue

                # Downstream alignment verification: v1 -> s1 and v2 -> s2
                s1 = children[v1]
                s2 = children[v2]
                c_s1 = coords[s1, 1:] * scale
                c_s2 = coords[s2, 1:] * scale
                v_down1 = c_s1 - c_v1
                v_down2 = c_s2 - c_v2
                d_d1 = np.linalg.norm(v_down1)
                d_d2 = np.linalg.norm(v_down2)

                if d_d1 > 0.2 and d_d2 > 0.2:
                    down_cur = float(np.dot(v_out1, v_down1)/(d_out1*d_d1) + np.dot(v_out2, v_down2)/(d_out2*d_d2))
                    down_swap = float(np.dot(v_swap2, v_down1)/(d_s2*d_d1) + np.dot(v_swap1, v_down2)/(d_s1*d_d2))
                    # Downstream alignment must not suffer severely
                    if down_swap < down_cur - 0.40:
                        continue

                # Morphological descriptor check if enabled
                if use_desc and descriptors is not None and len(descriptors) > max(u1, u2, v1, v2):
                    norm_u1 = np.linalg.norm(descriptors[u1])
                    norm_u2 = np.linalg.norm(descriptors[u2])
                    norm_v1 = np.linalg.norm(descriptors[v1])
                    norm_v2 = np.linalg.norm(descriptors[v2])
                    if min(norm_u1, norm_u2, norm_v1, norm_v2) > 0:
                        sim_cur = float(np.dot(descriptors[u1], descriptors[v1]) / (norm_u1 * norm_v1) +
                                        np.dot(descriptors[u2], descriptors[v2]) / (norm_u2 * norm_v2))
                        sim_swap = float(np.dot(descriptors[u1], descriptors[v2]) / (norm_u1 * norm_v2) +
                                         np.dot(descriptors[u2], descriptors[v1]) / (norm_u2 * norm_v1))
                        if sim_swap - sim_cur < min_desc_delta:
                            continue

                # Confirmed crossover untangling swap
                swaps_performed.append(((u1, v1), (u1, v2), (u2, v2), (u2, v1)))
                swapped_u.add(u1)
                swapped_u.add(u2)
                swapped_v.add(v1)
                swapped_v.add(v2)
                break

    if not swaps_performed:
        return edges, {"crossovers_inspected": crossovers_inspected, "crossovers_untangled": 0}

    # Apply edge replacements
    old_edges_set = set(map(tuple, edges))
    for (old1, new1, old2, new2) in swaps_performed:
        if old1 in old_edges_set and old2 in old_edges_set:
            old_edges_set.remove(old1)
            old_edges_set.remove(old2)
            old_edges_set.add(new1)
            old_edges_set.add(new2)

    new_edges = np.asarray(sorted(old_edges_set), dtype=edges.dtype)
    return new_edges, {
        "crossovers_inspected": crossovers_inspected,
        "crossovers_untangled": len(swaps_performed),
    }


def prune_spurious_cytokinesis_forks(coords, edges, scale, descriptors, pr_cfg):
    """Prunes false positive divisions violating eukaryotic cytokinesis angular divergence and spatial symmetry."""
    if not pr_cfg.get("enabled", False):
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.20)
    min_ratio = pr_cfg.get("min_ratio", 0.15)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)
    persistence_guard = pr_cfg.get("persistence_guard", True)

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for u, v in edges:
        out_edges[int(u)].append(int(v))
        in_edges[int(v)].append(int(u))

    divs = {u: vs for u, vs in out_edges.items() if len(vs) == 2}
    if not divs:
        return edges, {"forks_inspected": 0, "forks_pruned": 0}

    def forward_len(start):
        c = start
        l = 1
        while out_edges[c]:
            c = out_edges[c][0]
            l += 1
        return l

    parents = {int(v): int(u[0]) for v, u in in_edges.items() if len(u) == 1}

    edges_to_remove = set()
    for m, daughters in divs.items():
        d1, d2 = daughters
        vm = coords[m, 1:] * scale
        v1 = coords[d1, 1:] * scale - vm
        v2 = coords[d2, 1:] * scale - vm
        d_v1 = np.linalg.norm(v1)
        d_v2 = np.linalg.norm(v2)

        if d_v1 < 1e-6 or d_v2 < 1e-6:
            continue

        cos_angle = float(np.dot(v1, v2) / (d_v1 * d_v2))
        ratio = float(min(d_v1, d_v2) / max(d_v1, d_v2))
        max_d = float(max(d_v1, d_v2))

        # Check violation of cytokinesis physics
        is_acute = (cos_angle > cos_thresh)
        is_asymmetric = (ratio < min_ratio and max_d > min_max_dist_um)

        if is_acute or is_asymmetric:
            if persistence_guard:
                fwd1 = forward_len(d1)
                fwd2 = forward_len(d2)
                if fwd1 != fwd2:
                    if fwd1 > fwd2:
                        edges_to_remove.add((m, d2))
                    else:
                        edges_to_remove.add((m, d1))
                    continue

            if ratio < 0.15 and max_d > 3.0:
                if d_v1 < d_v2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                continue

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
    """Assembles candidate lineage graph through sequential label-free regularizations."""
    coords = anchor.coords.copy()
    edges = anchor.edges.copy()

    # Stage 1: Pure Symmetric Axial Regularizer (z-jitter)
    cfg_smoothing = variant_cfg.get("smoothing", {})
    smoothed_coords, audit_smoothing = apply_symmetric_axial_regularizer(
        coords, edges, scale, cfg_smoothing
    )

    # Stage 2: Momentum-Conserving Crossover Untangling
    cfg_xo = variant_cfg.get("crossover_untangling", {})
    untangled_edges, audit_crossover = untangle_crossover_swaps(
        smoothed_coords, edges, scale, descriptors, cfg_xo
    )

    # Stage 3: Cytokinesis Fork Pruning
    cfg_prune = variant_cfg.get("fork_pruning", {})
    retained_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, untangled_edges, scale, descriptors, cfg_prune
    )

    cand = Lineage(smoothed_coords, retained_edges)
    cand.validate()

    audit = {
        "smoothing": audit_smoothing,
        "crossover_untangling": audit_crossover,
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
        xo_cnt = audit["crossover_untangling"].get("crossovers_untangled", 0)
        pr_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, untangled={xo_cnt}, pruned={pr_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-025 on A1555."""
    cfg = config()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    run = Path(run)

    print(f"Starting Campaign 025 on A1555 across {len(names)} development datasets...", flush=True)
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
