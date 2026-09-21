"""One implementation for remote ablations and later notebook integration.

Inference never reads GT. GT enters only the separately invoked diagnostic
evaluator, after predictions have been written and their contract checked.
"""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from core.nextgen.contracts import (Lineage, check_manifest, load_candidates,
    load_reference, sha256, verify_file, write_submission)

ARMS = ("control", "mitosis_add", "hoct", "hoct_mitosis", "hoct_rescue", "full")


def check_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    if not 0 < config["solver"]["timeout_seconds"] <= 60:
        raise ValueError("SCIP time limit must be in (0,60] seconds")
    for section, names in {
        "candidates": ("neighbours", "max_edges"),
        "hoct": ("max_nodes_per_batch", "max_edges_per_batch"),
        "mitosis": ("max_children", "batch_size", "max_negatives_per_mother"),
        "rescue": ("max_nodes",), "training": ("epochs", "batch_size", "seed"),
        "solver": ("seed",),
    }.items():
        for name in names:
            value = config[section][name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Integer setting required: {section}.{name}")
    for section, names in {
        "candidates": ("neighbours", "radius_um", "max_edges", "prior_match_um"),
        "hoct": ("tile_um", "halo_um", "max_nodes_per_batch", "max_edges_per_batch", "mask_radius_um"),
        "mitosis": ("radius_um", "max_children", "batch_size", "max_negatives_per_mother"),
        "rescue": ("step_radius_um", "search_radius_um", "exclusion_um", "peak_sigma_um", "max_nodes", "min_snr"),
        "training": ("epochs", "batch_size", "learning_rate"),
    }.items():
        for name in names:
            if not np.isfinite(config[section][name]) or config[section][name] <= 0:
                raise ValueError(f"Invalid positive setting: {section}.{name}")
    for section, name in (("hoct", "mix_weight"), ("mitosis", "threshold"),
                          ("mitosis", "min_edge_probability"), ("rescue", "min_ncc"),
                          ("rescue", "ambiguity_ratio"), ("rescue", "max_fraction")):
        if not 0 <= config[section][name] <= 1:
            raise ValueError(f"Invalid probability/fraction: {section}.{name}")


def preflight(manifest, config, arm):
    from core.nextgen.images import ImageSource
    check_manifest(manifest)
    check_config(config)
    if arm not in ARMS:
        raise ValueError("Unknown ablation arm")
    verify_file(manifest["reference_csv"], manifest["reference_sha256"])
    if manifest["scope"] == "submission":
        from core.submission_contract import validate_csv
        validate_csv(Path(manifest["reference_csv"]), enforce_row_budget=True,
                     expected_datasets=[r["name"] for r in manifest["datasets"]])
    reference = load_reference(manifest["reference_csv"], [r["name"] for r in manifest["datasets"]])
    receipts = []
    for row in manifest["datasets"]:
        source = ImageSource(row["image"], row.get("image_key", "0"), row["scale_um"])
        reference[row["name"]].validate(source.shape)
        receipt = {"dataset": row["name"], "image_shape": source.shape, "scale_um": row["scale_um"]}
        if arm != "control":
            coords, edges = load_candidates(row)
            Lineage(coords, np.empty((0, 2), dtype=np.int64)).validate(source.shape)
            receipt.update(raw_nodes=len(coords), raw_edges=len(edges), candidates_sha256=row["candidates_sha256"])
        receipts.append(receipt)
    if arm not in {"control", "mitosis_add"}:
        import hoct._api as api
        import pyscipopt
        spec = manifest["hoct"]
        if spec.get("admissibility_reviewed") is not True or not spec.get("source_url"):
            raise ValueError("HOCT weights must be reviewed for competition admissibility before execution")
        verify_file(api.__file__, spec["api_sha256"])
        verify_file(spec["checkpoint"], spec["sha256"])
    if arm in {"mitosis_add", "hoct_mitosis", "full"}:
        verify_file(manifest["mitosis"]["checkpoint"], manifest["mitosis"]["sha256"])
    return {"inputs": receipts, "reference_sha256": manifest["reference_sha256"],
            "scope": manifest["scope"], "arm": arm, "status": "preflight_only_not_a_score"}


def infer_dataset(anchor, row, manifest, config, arm, device="cuda"):
    from core.nextgen.candidates import build_candidates
    from core.nextgen.images import ImageSource, measured_features
    from core.nextgen.hoct_bridge import score_edges, blend_probabilities
    from core.nextgen.solver import reference_divisions, solve_lineage
    if arm == "control":
        return Lineage(anchor.coords.copy(), anchor.edges.copy()), {"control": "unchanged reference nodes and edges"}
    source = ImageSource(row["image"], row.get("image_key", "0"), row["scale_um"])
    raw_coords, raw_edges = load_candidates(row)
    pairs, priors, report = build_candidates(anchor, raw_coords, raw_edges, source.scale, config["candidates"])
    if arm == "mitosis_add":
        from core.nextgen.mitosis import score_divisions, add_confirmed_pairs
        # Unknown priors are excluded from this arm, never promoted to invented
        # edge probabilities. Existing reference edges are preserved verbatim.
        known = np.isfinite(priors)
        divisions = score_divisions(anchor.coords, pairs[known], priors[known], source,
            manifest["mitosis"], row, config["mitosis"], device)
        graph, details = add_confirmed_pairs(anchor, divisions)
        graph.validate(source.shape)
        return graph, {**report, **details}
    features = measured_features(anchor.coords, source, config["hoct"]["mask_radius_um"])
    hoct_prob = score_edges(anchor.coords, pairs, features, source.scale, manifest["hoct"], config["hoct"], device)
    probabilities = blend_probabilities(hoct_prob, priors, config["hoct"]["mix_weight"])
    if arm in {"hoct_mitosis", "full"}:
        from core.nextgen.mitosis import score_divisions
        divisions = score_divisions(anchor.coords, pairs, probabilities, source, manifest["mitosis"], row, config["mitosis"], device)
    else:
        # HOCT-only cannot invent new mitoses; it can keep or reject reference pairs.
        divisions = reference_divisions(anchor.edges)
    graph, solve_report = solve_lineage(anchor.coords, pairs, probabilities, divisions, len(anchor.coords), config["solver"])
    report["first_solve"] = solve_report
    if arm in {"hoct_rescue", "full"}:
        from core.nextgen.rescue import propose_rescues
        proposals = propose_rescues(graph, source, config["rescue"])
        report["image_supported_rescue_proposals"] = proposals
        if proposals:
            # Candidate rescues are optional. The ILP must select both their links.
            coords = np.concatenate([anchor.coords, np.asarray([p["coord"] for p in proposals])])
            extra = [(u, v) for i, p in enumerate(proposals, len(anchor.coords))
                     for u, v in ((p["source"], i), (i, p["target"]))]
            pairs = np.concatenate([pairs, np.asarray(extra, dtype=np.int64)])
            priors = np.concatenate([priors, np.full(len(extra), np.nan)])
            if len(pairs) > config["candidates"]["max_edges"]:
                raise ValueError("Rescue candidates exceed memory budget")
            # Real image-derived features and new temporal scores, not midpoint confidence.
            new_features = measured_features(coords, source, config["hoct"]["mask_radius_um"])
            hp = score_edges(coords, pairs, new_features, source.scale, manifest["hoct"], config["hoct"], device)
            probabilities = blend_probabilities(hp, priors, config["hoct"]["mix_weight"])
            graph, solve_report = solve_lineage(coords, pairs, probabilities, divisions, len(anchor.coords), config["solver"])
            report["rescue_solve"] = solve_report
    graph.validate(source.shape)
    if not np.array_equal(graph.coords[:len(anchor.coords)], anchor.coords):
        raise ValueError("Reference detections moved or were pruned")
    report.update(nodes=len(graph.coords), edges=len(graph.edges), preserved_reference_nodes=len(anchor.coords))
    return graph, report


def evaluate_pairs(predictions, references, rows, metric_root):
    import sys
    import tracksdata as td
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    sys.path.insert(0, str(metric_root))
    from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
    by_arm = {"reference": [], "candidate": []}
    for row in rows:
        if not row.get("gt"):
            raise ValueError("Diagnostic evaluation requires an explicit public GT path")
        loaded = td.graph.InMemoryGraph.from_geff(row["gt"])
        gt = loaded[0] if isinstance(loaded, tuple) else loaded
        estimated = (GeffMetadata.read(row["gt"]).extra or {}).get("estimated_number_of_nodes")
        if estimated is None:
            raise ValueError("Official estimated_number_of_nodes missing; do not substitute annotated node count")
        for arm, graphs in (("reference", references), ("candidate", predictions)):
            graph = graphs[row["name"]]
            nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(graph.coords)}
            edges = [{"source_id": int(u), "target_id": int(v)} for u, v in graph.edges]
            frames, links = as_evaluator_inputs(nodes, edges, round_coordinates=False)
            pred = make_graph(frames, links)
            counts = evaluate(pred, gt, scale=tuple(row["scale_um"]))
            result = per_sample_metrics(counts, estimated, node_recall(pred, gt))
            by_arm[arm].append({**result, "dataset": row["name"], "group": row["group"], "split": row["split"], "counts": counts._asdict()})
    return {"rows": by_arm, "summary": {arm: summarise(values) for arm, values in by_arm.items()},
            "by_split": {split: {arm: summarise([r for r in values if r["split"] == split])
                for arm, values in by_arm.items()} for split in sorted({r["split"] for r in rows})},
            "by_group": {group: {arm: summarise([r for r in values if r["group"] == group])
                for arm, values in by_arm.items()} for group in sorted({r["group"] for r in rows})},
            "scope": "paired_public_diagnostic_detectors_full_fit_not_hidden_kaggle_score",
            "promoted": False}


def run_inference(manifest, config, arm, output, *, metric_root=None, device="cuda"):
    """Write isolated artifacts. Caller creates a NEW directory; never a Kaggle push."""
    output = Path(output)
    receipt = preflight(manifest, config, arm)
    reference = load_reference(manifest["reference_csv"], [r["name"] for r in manifest["datasets"]])
    predictions, reports = {}, {}
    for row in manifest["datasets"]:
        started = time.monotonic()
        name = row["name"]
        predictions[name], reports[name] = infer_dataset(reference[name], row, manifest, config, arm, device)
        reports[name]["seconds"] = time.monotonic() - started
        with (output / f"{name}.npz").open("xb") as handle:
            np.savez_compressed(handle, coords=predictions[name].coords, edges=predictions[name].edges)
        print(json.dumps({"dataset": name, **reports[name]}), flush=True)
    contract = write_submission(output / "candidate.csv", predictions, enforce_budget=manifest["scope"] == "submission")
    receipt.update(status="artifacts_valid_quality_unverified", contract=contract, datasets=reports,
        config=config, manifest=manifest, kaggle_score=None, promoted=False,
        versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy")})
    if manifest["scope"] == "diagnostic" and metric_root is not None:
        receipt["evaluation"] = evaluate_pairs(predictions, reference, manifest["datasets"], metric_root)
    if manifest["scope"] == "submission":
        # The name submission.csv only appears AFTER structural/cardinality validation.
        destination = output / "submission.csv"
        if destination.exists():
            raise FileExistsError(destination)
        (output / "candidate.csv").rename(destination)
    return receipt
