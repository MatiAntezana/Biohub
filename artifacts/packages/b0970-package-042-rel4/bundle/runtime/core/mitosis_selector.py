"""011: predicted-coordinate training, source-prefix calibration, target-prefix test.

The preparation job freezes label-free detections/features for NEW training/
calibration samples plus the eight historical development samples. Confirmation,
visible and010 oracle-edited graphs are never training data. Only this workflow
reads public GT; the association policy itself has no annotation interface.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import minimize

from core.artifacts import ROOT, code_manifest, code_hash, digest, dump
from core.mitosis_counterfactual import (verify_sources as verify_parent, metric_row,
    official_metrics, assert_baseline)
from core.mitosis_diagnostic import match_annotation
from core.mitosis_probe import standardize_training, class_weights, objective, probe_metrics
from core.mitosis_proposals import tree_fingerprint, label_pairs
from core.nextgen.association import (EDGE_NAMES, FORK_NAMES, candidate_tables,
    image_descriptors, context, edge_features, fork_features, linear_scores,
    score_tables, propose_transactions, apply_transactions)
from core.nextgen.contracts import Lineage, load_reference, verify_file
from core.protocol import PROTOCOL, resolve_dataset_row, assess

ADDITIONS = {"core/nextgen/association.py", "core/mitosis_selector.py",
    "configs/mitosis_selector.json", "tests/test_mitosis_selector.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "learned_selector_development_only_no_submission"
CORPUS_SCOPE = "selector_predicted_corpus_only_no_submission"


def config():
    return json.loads((ROOT / "configs/mitosis_selector.json").read_text())


def verify_sources(source, manifest):
    cfg = config()
    previous = {p: h for p, h in manifest.items() if p not in ADDITIONS}
    science = {p: h for p, h in previous.items() if p not in DRIVERS}
    identity = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    if identity != cfg["parent_science_sha256"] or source.name != cfg["source_run"]:
        raise ValueError("Selector requires exact010 science and pinned005 source")
    _, baseline = verify_parent(source, previous)
    # The010 report is evidence for the design ONLY, never a source of labels/edits.
    for relative, sha in cfg["source_artifacts"].items():
        verify_file(source.parent / relative, sha)
    return baseline


def split_corpus(plan, cfg):
    """Freeze dataset-level split from already archived public metadata, not scores.

    Per prefix: up to24 division-bearing samples and four zero-annotated-division
    samples. Three division-bearing + two zero-division samples are calibration.
    Zero annotated divisions does NOT mean every unlabeled fork is negative.
    """
    forbidden = set(PROTOCOL["previously_exposed"])
    for panel in plan["panels"].values():
        forbidden.update(panel)
    all_names = plan["all_datasets"]
    if len(set(all_names)) != len(all_names) or plan["panels"]["development"] != PROTOCOL["development"]:
        raise ValueError("Dataset plan drift")
    result = {}
    rank = lambda n: hashlib.sha256((cfg["split_seed"]+n).encode()).hexdigest()
    for group in PROTOCOL["group_prefixes"]:
        eligible = [n for n in all_names if n.startswith(group+"_") and n not in forbidden]
        positive = sorted((n for n in eligible if plan["public_info"][n]["divisions"] > 0),
            key=lambda n: (-plan["public_info"][n]["divisions"], rank(n)))[:cfg["max_positive_samples_per_group"]]
        zero = sorted((n for n in eligible if plan["public_info"][n]["divisions"] == 0), key=rank)[:cfg["zero_division_samples_per_group"]]
        if (len(positive) < cfg["calibration_positive_samples"]+cfg["minimum_positive_examples"]
                or len(zero) != cfg["zero_division_samples_per_group"]):
            raise ValueError("Insufficient unexposed training/calibration samples")
        # Re-rank selected positives for calibration, avoiding a richest-only fit.
        positive, zero = sorted(positive, key=rank), sorted(zero, key=rank)
        p, z = cfg["calibration_positive_samples"], cfg["calibration_zero_samples"]
        result[group] = dict(fit=sorted(positive[p:]+zero[z:]), calibration=sorted(positive[:p]+zero[:z]))
    used = [n for g in result.values() for names in g.values() for n in names]
    if len(used) != len(set(used)) or set(used) & forbidden:
        raise ValueError("Training/calibration overlap with reserved panels")
    return result


def save_evidence(path, anchor, source, cfg):
    edges, forks = candidate_tables(anchor, source.scale, cfg)
    descriptors, stats = image_descriptors(anchor, source, cfg)
    with path.open("xb") as handle:
        np.savez_compressed(handle, coords=anchor.coords, reference_edges=anchor.edges,
            edge_ids=edges, fork_ids=forks, descriptors=descriptors, stats=stats,
            scale=source.scale, shape=np.asarray(source.shape, np.int64))
    return dict(sha256=digest(path), nodes=len(anchor.coords), reference_edges=len(anchor.edges),
        proposed_edges=len(edges), proposed_forks=len(forks), shape=list(source.shape),
        scale_um=source.scale.tolist(), image=str(source.array.store),
        image_identity_scope="path_and_metadata_plus_hashed_measured_patches_not_full_image_digest")


def open_evidence(folder, name, manifest):
    if name not in manifest["datasets"]:
        raise ValueError("Dataset outside frozen selector corpus")
    path = folder / (name + ".evidence.npz")
    info = manifest["datasets"][name]
    verify_file(path, info["sha256"])
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    expected = {"coords", "reference_edges", "edge_ids", "fork_ids", "descriptors", "stats", "scale", "shape"}
    if set(a) != expected or a["shape"].tolist() != info["shape"] or a["scale"].tolist() != info["scale_um"]:
        raise ValueError("Corpus feature schema/metadata drift")
    graph = Lineage(a["coords"], a["reference_edges"]).validate(a["shape"])
    if len(graph.coords) != info["nodes"] or len(graph.edges) != info["reference_edges"]:
        raise ValueError("Corpus lineage count drift")
    for k, width in (("edge_ids", 2), ("fork_ids", 3)):
        ids = a[k]
        if (ids.ndim != 2 or ids.shape[1] != width or not np.issubdtype(ids.dtype, np.integer)
                or (ids < 0).any() or (ids >= len(graph.coords)).any()
                or not (graph.coords[ids[:, 1:], 0] == graph.coords[ids[:, :1], 0]+1).all()):
            raise ValueError("Invalid corpus proposal IDs")
    ctx = context(graph, a["scale"], a["descriptors"], a["stats"])
    return graph, a["edge_ids"], a["fork_ids"], ctx


def prepare_job(args, run, source):
    from core.nextgen.images import ImageSource
    from ops.workflow import cache_panel
    verify_sources(source, code_manifest())
    cfg = config()
    plan = json.loads((source / "plan.json").read_text())
    splits = split_corpus(plan, cfg)
    if shutil.disk_usage(run).free < cfg["minimum_free_disk_gib"] * 1024**3:
        raise RuntimeError("Selector corpus needs24 GiB free; do not remove prior runs")
    training_names = sorted(n for g in splits.values() for names in g.values() for n in names)
    dump(run / "split.json", dict(splits=splits, development=PROTOCOL["development"], recipe=cfg))
    # Actual predicted coordinates, NOT GT coordinates perturbed into pseudo-inputs.
    cache = run / "cache-training"
    cache_panel(training_names, plan["data_root"], Path(args.artifact_root)/"assets/champion", cache)
    training = load_reference(cache / "reference.csv", training_names)
    development = load_reference(source / "cache-development/reference.csv", PROTOCOL["development"])
    datasets = {}
    for name in sorted(training.keys() | development.keys()):
        row = resolve_dataset_row(name, plan["data_root"], split="test", gt=False)
        image = ImageSource(row["image"], "0", row["scale_um"])
        graph = training[name] if name in training else development[name]
        datasets[name] = save_evidence(run/(name+".evidence.npz"), graph, image, cfg)
        datasets[name].update(image=row["image"], group=row["group"],
            panel="development" if name in development else "training_or_calibration")
        del image
        dump(run / "progress.json", dict(completed=sorted(datasets), total=len(training)+len(development)))
        print(json.dumps(dict(phase="selector_corpus", dataset=name, nodes=len(graph.coords))), flush=True)
    verify_sources(source, code_manifest())
    corpus = dict(schema="selector_corpus_v1", disposition=CORPUS_SCOPE, code_hash=code_hash(),
        source_run=source.name, source_unchanged=True, recipe=cfg, data_root=plan["data_root"],
        splits=splits, development=sorted(PROTOCOL["development"]), datasets=datasets,
        confirmation_evaluated=False, gt_opened=False, model_trained=False,
        kaggle_submission_allowed=False, detector_full_fit=True)
    dump(run / "corpus.json", corpus)
    return dict(disposition=CORPUS_SCOPE, corpus_sha256=digest(run/"corpus.json"),
        source_unchanged=True, ready_for_selector=True, datasets=len(datasets),
        confirmation_evaluated=False, gt_opened=False, model_trained=False,
        kaggle_submission_allowed=False)


def verify_corpus(folder, manifest):
    receipt = json.loads((folder / "receipt.json").read_text())
    expected_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (receipt.get("task") != "prepare-selector" or receipt.get("status") != "completed"
            or receipt.get("code_hash") != expected_hash
            or receipt.get("result", {}).get("ready_for_selector") is not True):
        raise ValueError("Completed current-release selector corpus required")
    verify_file(folder/"corpus.json", receipt["result"]["corpus_sha256"])
    corpus = json.loads((folder / "corpus.json").read_text())
    source = folder.parent / config()["source_run"]
    baseline = verify_sources(source, manifest)
    plan = json.loads((source / "plan.json").read_text())
    splits = split_corpus(plan, config())
    expected = set(PROTOCOL["development"]) | {n for g in splits.values() for ns in g.values() for n in ns}
    if (corpus["schema"] != "selector_corpus_v1" or corpus["recipe"] != config()
            or corpus["code_hash"] != expected_hash or corpus["splits"] != splits
            or corpus["development"] != sorted(PROTOCOL["development"])
            or corpus["data_root"] != plan["data_root"] or set(corpus["datasets"]) != expected
            or corpus["source_run"] != source.name or corpus["disposition"] != CORPUS_SCOPE
            or corpus["gt_opened"] is not False or corpus["confirmation_evaluated"] is not False):
        raise ValueError("Selector corpus or partition provenance drift")
    for n, info in corpus["datasets"].items():
        if info["group"] != n.split("_", 1)[0]:
            raise ValueError("Corpus group drift")
        verify_file(folder/(n+".evidence.npz"), info["sha256"])
    return corpus, baseline


def edge_labels(ids, mapping, parents):
    inverse = {v: k for k, v in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Non-injective annotation matching")
    labels = []
    for u, v in ids:
        gu, gv = inverse.get(int(u)), inverse.get(int(v))
        labels.append(-1 if gu is None or gv not in parents else int(parents[gv] == gu))
    return np.asarray(labels, np.int8)


def fit_binary(x, y, names, cfg):
    if x.shape != (len(y), len(names)) or set(np.asarray(y).tolist()) != {0, 1}:
        raise ValueError("Both confirmed classes and the exact feature schema required")
    if (int((y == 1).sum()) < cfg["minimum_positive_examples"]
            or int((y == 0).sum()) < cfg["minimum_negative_examples"]):
        raise ValueError("Insufficient confirmed training examples; do not relabel unknowns")
    x, stats = standardize_training(x, cfg["std_floor"])
    weight = class_weights(y)
    fit = minimize(objective, np.zeros(x.shape[1]+1), args=(x, y, weight, cfg["l2"]),
        jac=True, method=cfg["optimizer"], options={k: cfg[k] for k in ("maxiter", "ftol", "gtol")})
    loss, gradient = objective(fit.x, x, y, weight, cfg["l2"])
    norm = float(np.max(np.abs(gradient)))
    if not fit.success or norm > cfg["max_gradient_inf"]:
        raise ValueError(f"Selector fit failed: {fit.message}; gradient_inf={norm}")
    return dict(feature_names=list(names), standardizer=stats, weights=fit.x[:-1].tolist(),
        intercept=float(fit.x[-1]), calibrated_probability=False,
        fit=dict(success=True, objective=loss, gradient_inf=norm, iterations=int(fit.nit),
                 positives=int((y == 1).sum()), negatives=int((y == 0).sum())))


def train_group(folder, corpus, group, cfg, out):
    names = corpus["splits"][group]["fit"]
    xs, ys = {k: [] for k in ("edge", "fork")}, {k: [] for k in ("edge", "fork")}
    evidence = {}
    for name in names:
        if corpus["datasets"][name]["group"] != group:
            raise ValueError("Cross-prefix training leakage")
        anchor, edges, forks, ctx = open_evidence(folder, name, corpus)
        gt = Path(corpus["data_root"]) / (name+".geff")
        before = tree_fingerprint(gt)
        mapping, _, parents, _ = match_annotation(anchor, gt, ctx["scale"])
        labels = dict(edge=edge_labels(edges, mapping, parents), fork=label_pairs(forks, mapping, parents))
        evidence[name] = dict(gt_sha256=before, labels={k: dict(Counter(map(str, v))) for k, v in labels.items()})
        for kind, ids, fn in (("edge", edges, edge_features), ("fork", forks, fork_features)):
            known = labels[kind] >= 0
            xs[kind].append(fn(ids[known], ctx))
            ys[kind].append(labels[kind][known])
        if tree_fingerprint(gt) != before:
            raise ValueError("Training annotations changed during fitting")
    model = dict(schema="association_selector_v1_diagnostic", recipe=cfg,
        training_groups=[group], training_datasets=names, selection_groups=[],
        selection_datasets=[], margin=None, scope=SCOPE, detector_full_fit=True)
    for kind, feature_names in (("edge", EDGE_NAMES), ("fork", FORK_NAMES)):
        x, y = np.concatenate(xs[kind]), np.concatenate(ys[kind])
        model[kind] = fit_binary(x, y, feature_names, cfg)
        model[kind]["training_only_metrics"] = probe_metrics(y, linear_scores(model[kind], x), .5)
    dump(out / ("fit-"+group+".json"), dict(model=model, annotation_evidence=evidence))
    return model, evidence


def check_model(model, data_group, *, calibration=False):
    groups = model.get("training_groups", [])
    if (model.get("schema") != "association_selector_v1_diagnostic" or model.get("scope") != SCOPE
            or model.get("recipe") != config() or len(groups) != 1
            or groups[0] not in PROTOCOL["group_prefixes"]):
        raise ValueError("Unsupported selector model/provenance")
    if calibration:
        if groups != [data_group] or model["selection_groups"]:
            raise ValueError("Calibration must use the unselected source-prefix fit")
    elif data_group in groups + model.get("selection_groups", []):
        raise ValueError("Evaluated prefix entered training or calibration")


def predict_proposals(folder, corpus, name, model, *, calibration=False):
    check_model(model, corpus["datasets"][name]["group"], calibration=calibration)
    anchor, edges, forks, ctx = open_evidence(folder, name, corpus)
    es, fs = score_tables(edges, forks, ctx, model, config())
    proposals = propose_transactions(ctx, es, fs, config())
    return anchor, proposals


def evaluate_panel(reference, candidate):
    if not reference or sorted(reference) != sorted(candidate):
        raise ValueError("Incomplete paired selector panel")
    metric = official_metrics()
    for name in reference:
        if reference[name]["num_pred_nodes"] != candidate[name]["num_pred_nodes"]:
            raise ValueError("Selector changed node cardinality")
    groups = sorted({r["group"] for r in reference.values()})
    rows = {"reference": list(reference.values()), "candidate": list(candidate.values())}
    summary = {k: metric.summarise(v) for k, v in rows.items()}
    grouped = {g: {k: metric.summarise([r for r in v if r["group"] == g]) for k, v in rows.items()} for g in groups}
    for summaries in [summary, *grouped.values()]:
        for s in summaries.values():
            if not np.isfinite([s[k] for k in ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard")]).all():
                raise ValueError("Undefined selector metric; no sample dropping")
    return dict(rows=rows, summary=summary, by_group=grouped, uses_gt_for_edits=False,
        scope="public_paired_diagnostic_full_fit_detector_not_kaggle_score", promoted=False)


def select_margin(results, cfg):
    """Four PREDECLARED margins plus abstention, evaluated on source calibration.

    No target-prefix scores enter this selection. A conservative tie chooses
    the larger margin. No admissible improvement means explicit abstention.
    """
    if sorted(results) != sorted(cfg["margin_grid"]):
        raise ValueError("Incomplete or unregistered margin grid")
    eligible = []
    for margin, result in results.items():
        e = result["evaluation"]
        gate = assess(e, "development")
        # The calibration improvement floor is separately predeclared.
        a, b = e["summary"]["reference"], e["summary"]["candidate"]
        reasons = [r for r in gate["reasons"] if not r.startswith("development: score improvement")]
        if (not reasons and not result["budget_rejected"]
                and b["score"]-a["score"] >= cfg["minimum_calibration_score_delta"]):
            eligible.append((b["score"], margin))
    return max(eligible)[1] if eligible else None


def evaluate_job(args, run, folder):
    cfg = config()
    corpus, archived = verify_corpus(folder, code_manifest())
    models, training_evidence = {}, {}
    # Fit BOTH source-only models before ANY calibration or development evaluation.
    for group in PROTOCOL["group_prefixes"]:
        models[group], training_evidence[group] = train_group(folder, corpus, group, cfg, run)
        print(json.dumps(dict(phase="selector_fit", group=group,
            edge=models[group]["edge"]["fit"], fork=models[group]["fork"]["fit"])), flush=True)
    # Freeze all label-free calibration proposals before opening calibration GT.
    prediction_hashes = {}
    for group in PROTOCOL["group_prefixes"]:
        for name in corpus["splits"][group]["calibration"]:
            _, proposals = predict_proposals(folder, corpus, name, models[group], calibration=True)
            path = run/(name+".calibration-proposals.json")
            dump(path, dict(proposals=proposals, trained_on=group, uses_gt_for_edits=False))
            prediction_hashes[name] = digest(path)
            print(json.dumps(dict(phase="calibration_prediction", dataset=name, proposals=len(proposals))), flush=True)
    dump(run/"calibration-prediction-manifest.json", prediction_hashes)
    calibrations, gt_hashes = {}, {}
    for group in PROTOCOL["group_prefixes"]:
        references, candidates = {}, {m: {} for m in cfg["margin_grid"]}
        budgets = {m: False for m in cfg["margin_grid"]}
        audits = {m: {} for m in cfg["margin_grid"]}
        for name in corpus["splits"][group]["calibration"]:
            path = run/(name+".calibration-proposals.json")
            verify_file(path, prediction_hashes[name])
            proposals = json.loads(path.read_text())["proposals"]
            anchor, _, _, _ = open_evidence(folder, name, corpus)
            row = resolve_dataset_row(name, corpus["data_root"], split="calibration", gt=True)
            gt_hashes[name] = tree_fingerprint(Path(row["gt"]))
            references[name] = {**metric_row(anchor, row), "split": "calibration"}
            for margin in cfg["margin_grid"]:
                graph, audit = apply_transactions(anchor, proposals, margin, cfg)
                audits[margin][name] = audit
                budgets[margin] |= audit["status"] == "rejected_edit_budget"
                candidates[margin][name] = {**metric_row(graph, row), "split": "calibration"}
        results = {m: dict(evaluation=evaluate_panel(references, candidates[m]),
            budget_rejected=budgets[m], edits=audits[m]) for m in cfg["margin_grid"]}
        margin = select_margin(results, cfg)
        models[group].update(margin=margin, selection_groups=[group],
            selection_datasets=corpus["splits"][group]["calibration"],
            abstained_for_insufficient_calibration_evidence=margin is None)
        calibrations[group] = dict(selected_margin=margin, trials=results)
        dump(run/("calibration-"+group+".json"), calibrations[group])
        print(json.dumps(dict(phase="calibration_selected", group=group, margin=margin)), flush=True)
    # Freeze model bytes BEFORE target-prefix development prediction/evaluation.
    specs = {}
    for group, model in models.items():
        path = run/("selector-"+group+".json")
        dump(path, model)
        specs[group] = dict(path=path.name, sha256=digest(path), margin=model["margin"])
    dump(run/"models.json", specs)
    predicted, audits, predictions = {}, {}, {}
    for name in corpus["development"]:
        other = next(g for g in PROTOCOL["group_prefixes"] if g != corpus["datasets"][name]["group"])
        verify_file(run/specs[other]["path"], specs[other]["sha256"])
        model = json.loads((run/specs[other]["path"]).read_text())
        anchor, proposals = predict_proposals(folder, corpus, name, model)
        graph, audit = apply_transactions(anchor, proposals, model["margin"], cfg)
        path = run/(name+".prediction.npz")
        with path.open("xb") as handle:
            np.savez_compressed(handle, coords=graph.coords, edges=graph.edges)
        predictions[name] = dict(sha256=digest(path), model_sha256=specs[other]["sha256"], trained_on=other)
        audits[name] = audit
        dump(run/(name+".edits.json"), audit)
        print(json.dumps(dict(phase="development_prediction", dataset=name,
            selected=len(audit["selected"]), status=audit["status"])), flush=True)
    dump(run/"development-prediction-manifest.json", predictions)
    # Only now evaluate target GT. No parameter changes after this boundary.
    baseline = {}
    archive_rows = {r["dataset"]: r for r in archived["rows"]["reference"]}
    for name in corpus["development"]:
        path = run/(name+".prediction.npz")
        verify_file(path, predictions[name]["sha256"])
        with np.load(path, allow_pickle=False) as f:
            graph = Lineage(f["coords"], f["edges"]).validate()
        anchor, _, _, _ = open_evidence(folder, name, corpus)
        np.testing.assert_array_equal(anchor.coords, graph.coords)
        row = resolve_dataset_row(name, corpus["data_root"], split="development", gt=True)
        gt_hashes[name] = tree_fingerprint(Path(row["gt"]))
        baseline[name] = metric_row(anchor, row)
        assert_baseline(baseline[name], archive_rows[name], 1e-10)
        predicted[name] = metric_row(graph, row)
        print(json.dumps(dict(phase="development_evaluation", dataset=name, baseline_reproduced=True)), flush=True)
    evaluation = evaluate_panel(baseline, predicted)
    gate = assess(evaluation, "development")
    if any(a["status"] == "rejected_edit_budget" for a in audits.values()):
        gate["passed"] = False
        gate["reasons"].append("At least one development sample rejected the edit budget")
    for group in training_evidence.values():
        gt_hashes.update({name: info["gt_sha256"] for name, info in group.items()})
    for name, sha in gt_hashes.items():
        if tree_fingerprint(Path(corpus["data_root"])/(name+".geff")) != sha:
            raise ValueError("Source annotations changed during selector evaluation")
    for group, spec in specs.items():
        verify_file(run/spec["path"], spec["sha256"])
    verify_corpus(folder, code_manifest())
    result = dict(disposition=SCOPE, source_corpus=folder.name, source_unchanged=True,
        evaluation=evaluation, development_gate=gate, models=specs, edits=audits,
        calibration_margins={g: m["margin"] for g, m in models.items()}, gt_sha256=gt_hashes,
        model_trained=True, threshold_selected=True, threshold_selection_scope="source_prefix_calibration_only",
        deployable_policy_evaluated=True, uses_gt_for_edits=False, tracking_evaluated=True,
        graph_exported=True, graph_export_scope="diagnostic_predictions_not_submission_csv",
        confirmation_evaluated=False, visible_evaluated=False, promoted=False,
        kaggle_submission_allowed=False, kaggle_score=None,
        next_action="designer_review_only" if gate["passed"] else "rejected_no_kaggle_submission")
    dump(run/"selector.json", result)
    return {k: result[k] for k in ("disposition", "source_unchanged", "development_gate", "model_trained",
        "threshold_selected", "threshold_selection_scope", "deployable_policy_evaluated",
        "uses_gt_for_edits", "tracking_evaluated", "graph_exported", "confirmation_evaluated",
        "visible_evaluated", "promoted", "kaggle_submission_allowed", "next_action")} | {
        "selector_sha256": digest(run/"selector.json")}
