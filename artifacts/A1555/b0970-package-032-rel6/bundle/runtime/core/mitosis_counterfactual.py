"""010: GT-contaminated evaluator experiment, NOT an inference algorithm.

Only in-memory edge interventions on frozen development detections. Never export
the counterfactual lineages or promote their scores. An exact fork under full
matching need not be a true positive under official windowed division matching.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from core.artifacts import ROOT, clean, code_manifest, digest, dump
from core.evaluation import metric_root
from core.mitosis_diagnostic import match_annotation
from core.mitosis_proposals import (verify_sources as verify_proposal_sources,
    topology_maps, edit_footprint, tree_fingerprint)
from core.nextgen.contracts import Lineage, load_reference, verify_file
from core.protocol import PROTOCOL, resolve_dataset_row

SCOPE = "gt_counterfactual_only_no_submission"
ADDITIONS = {"core/mitosis_counterfactual.py", "configs/mitosis_counterfactual.json",
    "tests/test_mitosis_counterfactual.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCENARIOS = ("covered_nonpresent", "covered_unique_top1_nonpresent", "matched_nonpresent")
SAFETY = dict(uses_gt_for_edits=True, confirmation_evaluated=False,
    tracking_evaluated=True, counterfactual_graphs_modified_in_memory=True,
    source_graph_modified=False, graph_exported=False, model_trained=False,
    threshold_selected=False, deployable_policy_evaluated=False, promoted=False,
    kaggle_submission_allowed=False, kaggle_score=None)


def config():
    cfg = json.loads((ROOT / "configs/mitosis_counterfactual.json").read_text())
    if tuple(cfg["scenarios"]) != SCENARIOS or cfg["scope"] != SCOPE:
        raise ValueError("Fixed counterfactual scenarios and non-publication scope required")
    return cfg


def verify_sources(source, manifest):
    cfg = config()
    previous = {p: h for p, h in manifest.items() if p not in ADDITIONS}
    science = {p: h for p, h in previous.items() if p not in DRIVERS}
    identity = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    if identity != cfg["parent_science_sha256"] or source.name != cfg["source_run"]:
        raise ValueError("Counterfactual requires exact 009 science and pinned 005 source")
    verify_proposal_sources(source, previous)
    for name, files in cfg["source_sha256"].items():
        for relative, sha in files.items():
            verify_file(source.parent / name / relative, sha)
    folder = source.parent / cfg["proposal_run"]
    receipt = json.loads((folder / "receipt.json").read_text())
    report = json.loads((folder / "proposals.json").read_text())
    if (receipt["status"] != "completed" or receipt["task"] != "propose-mitosis"
            or receipt["source_run"] != source.name
            or report["disposition"] != "proposal_assay_only_no_submission"
            or report["source_unchanged"] is not True
            or any(report[k] is not False for k in ("confirmation_evaluated",
                "tracking_evaluated", "graph_modified", "threshold_selected"))):
        raise ValueError("Expected completed, unmodified proposal-only 009")
    identities = json.loads((folder / "prediction-manifest.json").read_text())
    if (identities != report["prediction_sha256"]
            or sorted(report["datasets"]) != sorted(PROTOCOL["development"])
            or sorted(identities) != sorted(PROTOCOL["development"])):
        raise ValueError("Proposal development panel/identities changed")
    for name, sha in identities.items():
        verify_file(folder / (name + ".proposals.npz"), sha)
    baseline = json.loads((source.parent / cfg["baseline_run"] / "development/result.json").read_text())
    if baseline["reference_sha256"] != digest(source / "cache-development/reference.csv"):
        raise ValueError("Archived baseline uses another reference")
    return report, baseline["evaluation"]


def oracle_copy(anchor, triples):
    """Atomic GT intervention. Keep every coordinate; reject conflicting requests."""
    anchor.validate()
    requests = np.asarray(triples)
    if requests.size == 0:
        requests = np.empty((0, 3), np.int64)
    if (requests.ndim != 2 or requests.shape[1] != 3
            or not np.issubdtype(requests.dtype, np.integer)
            or (requests < 0).any() or (requests >= len(anchor.coords)).any()):
        raise ValueError("Invalid counterfactual node IDs")
    children, parents = topology_maps(anchor)
    desired, owners, additions, removals = {}, {}, set(), set()
    for mother, a, b in requests.tolist():
        if a == b or not (anchor.coords[[a, b], 0] == anchor.coords[mother, 0] + 1).all():
            raise ValueError("Counterfactual must be a consecutive two-child fork")
        pair = {a, b}
        if mother in desired or any(v in owners and owners[v] != mother for v in pair):
            raise ValueError("Conflicting counterfactual requests; never choose a winner")
        desired[mother] = pair
        owners.update({v: mother for v in pair})
        footprint = edit_footprint((mother, a, b), children, parents)
        additions.update(map(tuple, footprint["add_edges"]))
        removals.update(map(tuple, footprint["remove_edges"]))
    if additions & removals:
        raise ValueError("Conflicting addition/removal requests")
    original = set(map(tuple, anchor.edges))
    if not removals <= original or additions & original:
        raise ValueError("Invalid edit transaction")
    retained = [tuple(e) for e in anchor.edges if tuple(e) not in removals]
    result = Lineage(anchor.coords.copy(),
        np.asarray(retained + sorted(additions), np.int64).reshape(-1, 2)).validate()
    final_children, _ = topology_maps(result)
    if any(final_children.get(mother, set()) != pair for mother, pair in desired.items()):
        raise ValueError("Requested fork lost in simultaneous transaction")
    return result, {"requested_forks": len(requests), "added_edges": sorted(additions),
        "removed_edges": sorted(removals), "node_count": len(result.coords),
        "edge_count": len(result.edges), "edge_count_delta": len(additions)-len(removals)}


def audited_forks(anchor, info, gt_path):
    if tree_fingerprint(gt_path) != info["gt_sha256"]:
        raise ValueError("Development GT differs from the store audited in 009")
    mapping, children, _, coords = match_annotation(anchor, gt_path, info["scale_um"])
    forks = info["exact_gt_forks"]
    expected = [[int(m), *sorted(map(int, ds))] for m, ds in sorted(children.items()) if len(ds) == 2]
    if [f["gt_ids"] for f in forks] != expected:
        raise ValueError("GT fork identities changed since 009")
    pred_children, parents = topology_maps(anchor)
    reports = []
    for fork in forks:
        ids = [mapping.get(i) for i in fork["gt_ids"]]
        if ids != fork["anchor_ids"]:
            raise ValueError("Full matching changed since 009")
        row = dict(fork)
        if all(i is not None for i in ids):
            footprint = edit_footprint(ids, pred_children, parents)
            if "edit_footprint" in fork and clean(footprint) != fork["edit_footprint"]:
                raise ValueError("Archived edit footprint differs from reference")
            gt_coords = coords[fork["gt_ids"][0]]
            predicted = anchor.coords[ids]
            scale = np.asarray(info["scale_um"], float)
            row.update(endpoint_residual_um=np.linalg.norm((predicted[:, 1:]-gt_coords[:, 1:])*scale, axis=1).tolist(),
                gt_daughter_distances_um=np.linalg.norm((gt_coords[1:, 1:]-gt_coords[0, 1:])*scale, axis=1).tolist(),
                predicted_daughter_distances_um=np.linalg.norm((predicted[1:, 1:]-predicted[0, 1:])*scale, axis=1).tolist())
        reports.append(row)
    if tree_fingerprint(gt_path) != info["gt_sha256"]:
        raise ValueError("Development GT changed during audit")
    return reports


def scenario_forks(forks):
    matched = [f for f in forks if f["reason"] != "nonconsecutive_gt_fork"
        and all(i is not None for i in f["anchor_ids"])
        and f["edit_footprint"]["kind"] != "already_present"]
    covered = [f for f in matched if f["covered"]]
    return dict(covered_nonpresent=covered,
        covered_unique_top1_nonpresent=[f for f in covered if f["rank_worst"] == 1],
        matched_nonpresent=matched)


def official_metrics():
    root = metric_root().resolve()
    sys.path.insert(0, str(root))
    from tracking_cellmot import metrics
    if not Path(metrics.__file__).resolve().is_relative_to(root):
        raise ValueError("Official metric import came from outside the frozen vendor")
    return metrics


def metric_row(graph, row):
    """Same adapter/metadata as evaluate_pairs, with fresh mutable graphs per call."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph
    metric = official_metrics()
    gt = load_graph(row["gt"])
    estimated = (GeffMetadata.read(row["gt"]).extra or {}).get("estimated_number_of_nodes")
    if estimated is None or not np.isfinite(estimated) or estimated <= 0:
        raise ValueError("Valid official estimated_number_of_nodes required")
    graph.validate()
    nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(graph.coords)}
    edges = [{"source_id": int(u), "target_id": int(v)} for u, v in graph.edges]
    pred = make_graph(*as_evaluator_inputs(nodes, edges, round_coordinates=False))
    counts = metric.evaluate(pred, gt, scale=tuple(row["scale_um"]))
    values = metric.per_sample_metrics(counts, estimated, metric.node_recall(pred, gt))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Undefined per-sample official metric; stop, never drop sample")
    return {**values, "dataset": row["name"], "group": row["group"],
        "split": "development", "counts": counts._asdict()}


def assert_baseline(actual, archived, atol):
    # Historical evaluator called its development rows 'confirmation' to enforce
    # opposite-prefix loading. Compare dataset IDs/metrics, not that split label.
    if actual["dataset"] != archived["dataset"] or actual["group"] != archived["group"]:
        raise ValueError("Baseline dataset/group differs")
    if actual["counts"] != archived["counts"]:
        raise ValueError("Baseline counts differ from 005; no oracle evaluation permitted")
    keys = ("node_recall", "total_node_ratio", "edge_jaccard", "adj_edge_jaccard")
    a, b = [actual[k] for k in keys], [archived[k] for k in keys]
    if not np.isfinite(a + b).all() or not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError("Baseline metrics differ from 005")


def compare_rows(baseline, candidate):
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("Incomplete paired panel")
    metric = official_metrics()
    groups = sorted({r["group"] for r in baseline.values()})
    def compare(names):
        a, b = (metric.summarise([rows[n] for n in sorted(names)]) for rows in (baseline, candidate))
        required = ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard")
        if not np.isfinite([s[k] for s in (a, b) for k in required]).all():
            raise ValueError("Undefined aggregate metric; no partial denominator")
        if any(candidate[n]["num_pred_nodes"] != baseline[n]["num_pred_nodes"] for n in names):
            raise ValueError("Counterfactual changed cardinality")
        return {"reference": a, "counterfactual": b,
            "delta": {k: b[k]-a[k] for k in required},
            "count_delta": {k: sum(candidate[n][k]-baseline[n][k] for n in names)
                for k in metric.COUNT_COLUMNS}, "uses_gt_for_edits": True, "promoted": False}
    return {"panel": compare(baseline), "by_group": {
        g: compare([n for n in baseline if baseline[n]["group"] == g]) for g in groups}}


def counterfactual_job(args, run, source):
    report, archived = verify_sources(source, code_manifest())
    cfg = config()
    plan = json.loads((source / "plan.json").read_text())
    names = sorted(PROTOCOL["development"])
    if plan["panels"]["development"] != PROTOCOL["development"]:
        raise ValueError("Development panel drift")
    reference = load_reference(source / "cache-development/reference.csv", names)
    prior = {r["dataset"]: r for r in archived["rows"]["reference"]}
    if sorted(prior) != names:
        raise ValueError("Archived baseline panel differs")
    baseline, rows = {}, {}
    # All eight baseline results must reproduce 005 before ANY intervention.
    for name in names:
        info = report["datasets"][name]
        row = resolve_dataset_row(name, plan["data_root"], split="development", gt=True)
        if row["scale_um"] != info["scale_um"] or row["group"] != info["evaluated_data_group"]:
            raise ValueError("Dataset spacing/group drift")
        reference[name].validate(info["shape"])
        if tree_fingerprint(Path(row["gt"])) != info["gt_sha256"]:
            raise ValueError("GT differs from 009")
        baseline[name] = metric_row(reference[name], row)
        assert_baseline(baseline[name], prior[name], cfg["baseline_float_atol"])
        rows[name] = row
        dump(run / (name + ".baseline.json"), baseline[name])
        print(json.dumps({"phase": "baseline_parity", "dataset": name, "passed": True}), flush=True)
    dump(run / "baseline-parity.json", {"passed": True, "datasets": names,
        "comparison": compare_rows(baseline, baseline), "source_baseline_run": cfg["baseline_run"]})
    scenarios = {s: {} for s in SCENARIOS}
    audits, singles = {}, []
    for name in names:
        anchor, info = reference[name], report["datasets"][name]
        frozen_c, frozen_e = anchor.coords.copy(), anchor.edges.copy()
        forks = audited_forks(anchor, info, Path(rows[name]["gt"]))
        selected = scenario_forks(forks)
        cache = {anchor.edges.tobytes(): baseline[name]}
        local = {"dataset": name, "forks": forks, "scenarios": {}, "singles": [], **SAFETY}
        def evaluate(fork_rows):
            graph, edits = oracle_copy(anchor, [f["anchor_ids"] for f in fork_rows])
            key = graph.edges.tobytes()
            reused = key in cache
            if not reused:
                cache[key] = metric_row(graph, rows[name])
            return {"gt_mother_ids": [f["gt_ids"][0] for f in fork_rows],
                "edits": edits, "metrics": cache[key], "identical_graph_metrics_reused": reused}
        for scenario in SCENARIOS:
            item = evaluate(selected[scenario])
            local["scenarios"][scenario] = item
            scenarios[scenario][name] = item["metrics"]
        for fork in selected["matched_nonpresent"]:
            item = {**evaluate([fork]), "covered_in_009": fork["covered"],
                "rank_worst": fork.get("rank_worst"), "dataset": name}
            local["singles"].append(item)
            singles.append(item)
        if not np.array_equal(anchor.coords, frozen_c) or not np.array_equal(anchor.edges, frozen_e):
            raise ValueError("Source lineage mutated")
        if tree_fingerprint(Path(rows[name]["gt"])) != info["gt_sha256"]:
            raise ValueError("GT changed during counterfactual evaluation")
        audits[name] = local
        dump(run / (name + ".counterfactual.json"), local)
        print(json.dumps({"phase": "counterfactual", "dataset": name,
            "covered_nonpresent": len(selected["covered_nonpresent"]),
            "distinct_graph_evaluations": len(cache)}), flush=True)
    for item in singles:
        # Paired full-panel delta for ONE edit, never sum these deltas as an oracle.
        item["comparison"] = compare_rows(baseline, {**baseline, item["dataset"]: item["metrics"]})
    comparisons = {s: compare_rows(baseline, values) for s, values in scenarios.items()}
    verify_sources(source, code_manifest())
    for name in names:
        if tree_fingerprint(Path(rows[name]["gt"])) != report["datasets"][name]["gt_sha256"]:
            raise ValueError("GT changed before final receipt")
    result = {"disposition": SCOPE, "source_run": source.name, "proposal_run": cfg["proposal_run"],
        "recipe": cfg, "source_unchanged": True, "baseline_reproduced": True,
        "datasets": audits, "scenarios": comparisons, "single_fork_effects": singles,
        "scope": "development_gt_interventions_not_model_performance_not_a_rigorous_upper_bound",
        **SAFETY}
    dump(run / "counterfactual.json", result)
    return {"disposition": SCOPE, "counterfactual_sha256": digest(run / "counterfactual.json"),
        "source_unchanged": True, "baseline_reproduced": True, **SAFETY}
