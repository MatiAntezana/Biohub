"""013 event-balanced preference learning; diagnostic decisions, never graph edits.

Public annotations enter training and POST-prediction diagnostics only. Models,
features and calibration stay source-prefix-only. Reused panels/full-fit detector
are not an independent holdout. Unknowns never become negative training examples.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from core.artifacts import ROOT, code_manifest, code_hash, digest, dump
from core import repair_ranking as old, mitosis_selector as source
from core.nextgen import event_ranker as p, repair_ranker as r
from core.nextgen.contracts import verify_file

ADDITIONS = {"core/event_ranking.py", "core/nextgen/event_ranker.py",
             "configs/event_ranker.json", "tests/test_event_ranking.py", "ops/audit_event_inputs.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "event_ranking_assay_only_no_submission"
UNKNOWN, MIXED, NOOP = 2, 0, 3


def config():
    return json.loads((ROOT/"configs/event_ranker.json").read_text())


def parent_manifest(current, archived):
    if hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest() != config()["source_release"]:
        raise ValueError("Unverified archived012 code manifest")
    if ({k: v for k, v in current.items() if k not in ADDITIONS | DRIVERS}
            != {k: v for k, v in archived.items() if k not in DRIVERS}):
        raise ValueError("Unreviewed modification to science/configuration001-012")
    return archived


def audit_inputs(runs, manifest=None):
    """Only hashes/metadata; usable locally without annotations or fitting."""
    runs, cfg = Path(runs), config()
    folder = runs/cfg["source_run"]
    for name, sha in cfg["source_artifacts"].items():
        verify_file(folder/name, sha)
    receipt = json.loads((folder/"receipt.json").read_text())
    archived = parent_manifest(code_manifest() if manifest is None else manifest, receipt["code_manifest"])
    if (receipt["status"] != "completed" or receipt["task"] != "rank-repairs"
            or receipt["code_hash"] != cfg["source_release"] or receipt["node"] != cfg["node"]):
        raise ValueError("Expected completed pinned012 source on A1555")
    report = json.loads((folder/"ranking.json").read_text())
    if (report["next_action"] != "rejected_no_kaggle_submission"
            or report["source_unchanged"] is not True
            or any(report[k] is not False for k in
                   ("confirmation_evaluated", "visible_evaluated", "kaggle_submission_allowed"))):
        raise ValueError("Unsupported012 outcome")
    # Verify the historical manifest at each boundary, never rewrite a receipt
    # or pass a stripped current manifest as though it were an archived release.
    for relative, sha in old.config()["source_artifacts"].items():
        verify_file(runs/relative, sha)
    corpus_dir = runs/cfg["corpus_run"]
    receipt011 = json.loads((corpus_dir/"receipt.json").read_text())
    old.parent_manifest(archived, receipt011["code_manifest"])
    corpus = json.loads((corpus_dir/"corpus.json").read_text())
    if len(corpus["datasets"]) != 53 or len(corpus["development"]) != 8:
        raise ValueError("Incomplete011 evidence")
    for name, info in corpus["datasets"].items():
        verify_file(corpus_dir/(name+".evidence.npz"), info["sha256"])
    for group in source.PROTOCOL["group_prefixes"]:
        prefs = json.loads((folder/("preferences-"+group+".json")).read_text())
        verify_file(folder/prefs["tensor_path"], prefs["tensor_sha256"])
        if set(prefs["datasets"]) != set(corpus["splits"][group]["fit"]):
            raise ValueError("012 training annotations outside source fit")
        model = json.loads((folder/("ranker-"+group+".json")).read_text())
        if (model["training_groups"] != [group] or model["selection_groups"] != [group]
                or model["training_datasets"] != corpus["splits"][group]["fit"]
                or model["selection_datasets"] != corpus["splits"][group]["calibration"]):
            raise ValueError("012 comparator prefix leakage")
    return dict(passed=True, evidence_npz=53, preference_npz=2, source_release=cfg["source_release"],
        current_code_hash=code_hash(), gt_opened=False, historical005_chain_verified=False), corpus, receipt011["code_manifest"]


def verify_sources(folder, manifest):
    if folder.name != config()["source_run"]:
        raise ValueError("013 requires pinned b0970-ranking-012; do not rerun012")
    _, corpus, archived011 = audit_inputs(folder.parent, manifest)
    checked, _ = source.verify_corpus(folder.parent/config()["corpus_run"], archived011)
    if checked != corpus:
        raise ValueError("Corpus changed during verification")
    return corpus


def edge_truth(mapping, parents):
    inverse = {int(v): int(k) for k, v in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Non-injective annotation matching")
    def label(edge):
        a, b = (inverse.get(int(i)) for i in edge)
        return None if a is None or b not in parents else int(parents[b] == a)
    return label


def contrast_label(good, other, truth):
    """Label the net symmetric difference, cancelling SHARED edits first.

    No edited graph is materialized. Unknown shared changes cancel; unknown
    changes that differ still invalidate the comparison. Mixed trades stay mixed.
    """
    ga, gr, oa, ore = [set(map(tuple, es)) for es in
                      (good["add"], good["remove"], other["add"], other["remove"])]
    add, remove = (ga-oa) | (ore-gr), (gr-ore) | (oa-ga)
    if not add and not remove:
        return NOOP
    labels = [truth(e) for e in sorted(add)+sorted(remove)]
    if any(v is None for v in labels):
        return UNKNOWN
    delta = [2*v-1 for v in labels[:len(add)]] + [1-2*v for v in labels[len(add):]]
    return 1 if all(v == 1 for v in delta) else -1 if all(v == -1 for v in delta) else MIXED


def event_examples(name, forks, ctx, mapping, children, parents):
    """Compare each covered GT fork with existing competing hypotheses + keep.

    All alternatives use observed detections and the frozen011 universe. Repeated
    comparisons are not new biological events. The right sentinel means KEEP,
    not a fabricated graph node. Reference-correct forks remain useful examples.
    """
    ids = p.hypothesis_ids(forks, ctx)
    truth, rows, xs, counts, covered = edge_truth(mapping, parents), [], [], Counter(), 0
    for gt_m, gt_ds in sorted(children.items()):
        if len(gt_ds) != 2:
            continue
        g = [gt_m, *sorted(gt_ds)]
        if any(n not in mapping for n in g):
            counts["unmatched_gt_fork"] += 1
            continue
        target = (mapping[gt_m], *sorted(mapping[d] for d in gt_ds))
        local = ids[ids[:, 0] == target[0]]
        if not any(tuple(i) == target for i in local):
            counts["outside_frozen_pool"] += 1
            continue
        covered += 1
        good = p.transaction(target, ctx)
        good_x = r.features([good], ctx)[0]
        alternatives = [(tuple(i), p.transaction(i, ctx)) for i in local if tuple(i) != target]
        if (good["add"] or good["remove"]) and not any(
                not t["add"] and not t["remove"] for _, t in alternatives):
            alternatives.append(((target[0], -1, -1), p.reference(target[0])))
        usable = 0
        for key, other in alternatives:
            label = contrast_label(good, other, truth)
            counts[{1: "accepted", -1: "contradictory", 0: "mixed", 2: "unknown", 3: "same_graph"}[label]] += 1
            if label != 1:
                continue
            xs.append(good_x-r.features([other], ctx)[0])
            rows.append(dict(event=name+":"+str(target[0]), gt_mother=int(gt_m),
                preferred=list(target), alternative=list(key),
                reference_correct=not bool(good["add"] or good["remove"])))
            usable += 1
        counts["usable_events"] += bool(usable)
    return np.asarray(xs, float).reshape(-1, len(r.NAMES)), rows, dict(counts, covered_gt_forks=covered)


def collect_training(folder, corpus, group, run):
    cfg = config()
    with np.load(folder/("preferences-"+group+".npz"), allow_pickle=False) as f:
        x, y, keys = f["differences"], f["signs"], f["mother_keys"].tolist()
    if not set(k.rsplit(":", 1)[0] for k in keys) <= set(corpus["splits"][group]["fit"]):
        raise ValueError("Direct preferences outside source fit")
    strata = np.where(y == 1, "beneficial", "harmful").tolist()
    xs, ys, events, gt_hashes, datasets = [x], y.tolist(), [], {}, {}
    old_meta = json.loads((folder/("preferences-"+group+".json")).read_text())
    for name in corpus["splits"][group]["fit"]:
        if corpus["datasets"][name]["group"] != group:
            raise ValueError("Target prefix in fitting")
        anchor, _, forks, ctx = source.open_evidence(folder.parent/cfg["corpus_run"], name, corpus)
        gt = Path(corpus["data_root"])/(name+".geff")
        before = source.tree_fingerprint(gt)
        if before != old_meta["datasets"][name]["gt_sha256"]:
            raise ValueError("Training annotations differ from012")
        mapping, children, parents, _ = source.match_annotation(anchor, gt, ctx["scale"])
        extra, evidence, stats = event_examples(name, forks, ctx, mapping, children, parents)
        xs.append(extra)
        ys.extend([1.]*len(extra))
        keys.extend(e["event"] for e in evidence)
        strata.extend(["event"]*len(extra))
        events.extend(evidence)
        datasets[name], gt_hashes[name] = stats, before
        if source.tree_fingerprint(gt) != before:
            raise ValueError("Training annotations changed during013")
        print(json.dumps(dict(phase="event_preferences", dataset=name, **stats)), flush=True)
    data = dict(x=np.concatenate(xs), y=np.asarray(ys), keys=np.asarray(keys, str), strata=np.asarray(strata, str))
    summary = training_summary(data)
    path = run/("training-"+group+".npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    dump(run/("training-"+group+".json"), dict(summary=summary, datasets=datasets,
        events=events, direct_preferences_source=folder.name, tensor_sha256=digest(path),
        gt_sha256=gt_hashes, repeated_comparisons_not_independent_events=True))
    return data, summary, gt_hashes


def training_summary(data):
    result = {}
    for kind in ("beneficial", "harmful", "event"):
        mask = data["strata"] == kind
        keys = set(data["keys"][mask])
        result[kind] = dict(comparisons=int(mask.sum()), events=len(keys),
            volumes=len({k.rsplit(":", 1)[0] for k in keys}))
    result["unique_mothers_across_strata"] = len(set(data["keys"]))
    return result


def sufficient(summary, cfg):
    return (summary["beneficial"]["events"] >= cfg["minimum_beneficial_events"]
        and summary["harmful"]["events"] >= cfg["minimum_harmful_events"]
        and summary["event"]["events"] >= cfg["minimum_fork_events"]
        and summary["event"]["volumes"] >= cfg["minimum_fork_volumes"])


def event_weights(keys, strata, cfg):
    keys, strata = np.asarray(keys), np.asarray(strata)
    mass = cfg["stratum_mass"]
    if (keys.shape != strata.shape or keys.ndim != 1 or not len(keys)
            or set(strata) != set(mass) or not np.isclose(sum(mass.values()), 1.)
            or any(v <= 0 for v in mass.values())):
        raise ValueError("All three nonempty, event-balanced supervision strata required")
    weights = np.zeros(len(keys))
    for kind, amount in mass.items():
        mask = strata == kind
        counts = Counter(keys[mask])
        weights[mask] = [amount/(len(counts)*counts[k]) for k in keys[mask]]
    return weights


def preference_metrics(data, model):
    gain = r.scores(model, data["x"])
    result = {}
    for kind in config()["stratum_mass"]:
        take = data["strata"] == kind
        satisfied = gain[take]*data["y"][take] > 0
        keys = data["keys"][take]
        per_event = [float(satisfied[keys == k].mean()) for k in sorted(set(keys))]
        result[kind] = dict(comparisons=int(take.sum()), satisfied=int(satisfied.sum()),
            fraction=float(satisfied.mean()) if len(satisfied) else None,
            event_macro_fraction=float(np.mean(per_event)) if per_event else None)
    return result


def fit_event_ranker(data, cfg):
    x, y, strata, keys = (data[k] for k in ("x", "y", "strata", "keys"))
    if (x.shape != (len(y), len(r.NAMES)) or len(strata) != len(y)
            or not np.isfinite(x).all() or not np.isin(y, [-1, 1]).all()
            or not (y[strata == "harmful"] == -1).all()
            or not (y[strata != "harmful"] == 1).all()):
        raise ValueError("Invalid confirmed training preferences")
    weights = event_weights(keys, strata, cfg)
    rms = np.sqrt(np.sum(weights[:, None]*x*x, axis=0))
    active = rms >= cfg["std_floor"]
    scale = np.where(active, rms, 1.)
    signed = x/scale*active*y[:, None]
    fit = minimize(old.rank_objective, np.zeros(x.shape[1]), args=(signed, weights, cfg["l2"]),
        jac=True, method="L-BFGS-B", options={k: cfg[k] for k in ("maxiter", "ftol", "gtol")})
    loss, grad = old.rank_objective(fit.x, signed, weights, cfg["l2"])
    norm = float(np.max(np.abs(grad)))
    if not fit.success or not np.isfinite(loss) or not np.isfinite(norm) or norm > cfg["max_gradient_inf"]:
        raise ValueError(f"013 optimization failed: {fit.message}; gradient_inf={norm}")
    return dict(feature_names=list(r.NAMES), weights=fit.x.tolist(), intercept=0.,
        standardizer=dict(mean=[0.]*x.shape[1], scale=scale.tolist(), active=active.tolist()),
        fit=dict(objective=loss, gradient_inf=norm, iterations=int(fit.nit), fits=1,
            stratum_mass={k: float(weights[strata == k].sum()) for k in cfg["stratum_mass"]}),
        calibrated_probability=False)


def check_model(model, corpus, name, calibration=False):
    trained = model.get("training_groups", [])
    if (model.get("schema") != "event_ranker_013" or model.get("scope") != SCOPE
            or model.get("recipe") != config() or len(trained) != 1
            or trained[0] not in source.PROTOCOL["group_prefixes"]):
        raise ValueError("Invalid event model identity")
    group = trained[0]
    if model.get("training_datasets") != corpus["splits"][group]["fit"]:
        raise ValueError("Event fitting partition drift")
    if calibration:
        if (name not in corpus["splits"][group]["calibration"] or model.get("selection_groups")
                or model.get("selection_datasets") or model.get("diagnostic_margin") is not None):
            raise ValueError("Only unselected source calibration allowed")
    elif (name not in corpus["development"] or corpus["datasets"][name]["group"] == group
            or model.get("selection_groups") != [group]
            or model.get("selection_datasets") != corpus["splits"][group]["calibration"]):
        raise ValueError("Target prefix entered fitting/calibration")
    margin = model.get("diagnostic_margin")
    if margin is not None and (not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin < 0):
        raise ValueError("Invalid diagnostic margin")


def predict(folder, corpus, name, model, comparator, calibration=False):
    check_model(model, corpus, name, calibration)
    if comparator["training_groups"] != model["training_groups"]:
        raise ValueError("Comparator trained on wrong prefix")
    _, _, forks, ctx = source.open_evidence(folder.parent/config()["corpus_run"], name, corpus)
    ids, values, changed = p.score_universe(forks, ctx,
        [model["ranker"], comparator["ranker"]], config()["feature_batch_size"])
    return dict(ids=ids, scores=values, changed=changed)


def annotation_labels(ids, ctx, mapping, parents):
    truth = edge_truth(mapping, parents)
    return np.asarray([contrast_label(p.transaction(i, ctx), p.reference(i[0]), truth)
                       for i in ids], np.int8)


def fork_ranks(ids, values, changed, mapping, children):
    """Rank true forks against ALL alternatives, not just confirmed negatives.

    Include KEEP at score zero for new forks. Do not count an already-present
    fork and KEEP as two distinct hypotheses of the identical graph.
    """
    rows = []
    for mother, daughters in sorted(children.items()):
        if len(daughters) != 2:
            continue
        row = dict(gt_mother=int(mother), status="unmatched_endpoint")
        if all(k in mapping for k in [mother, *daughters]):
            target = (mapping[mother], *sorted(mapping[k] for k in daughters))
            choices = np.flatnonzero(ids[:, 0] == target[0])
            exact = [i for i in choices if tuple(ids[i]) == target]
            row.update(hypothesis=list(target), status="outside_frozen_pool")
            if exact:
                if len(exact) != 1:
                    raise ValueError("Duplicated covered fork")
                i = exact[0]
                row.update(status="covered", already_present=not bool(changed[i]), models={})
                for col, key in enumerate(("event013", "ranker012")):
                    other = values[choices[choices != i], col]
                    if changed[i] and not (~changed[choices]).any():
                        other = np.append(other, 0.)
                    worst = 1+int((other >= values[i, col]-1e-10).sum())
                    row["models"][key] = dict(gain=float(values[i, col]), rank_worst=worst,
                        unique_top1=worst == 1)
        rows.append(row)
    return rows


def summarize_tables(tables, col, margin):
    """Partial-label proposal metrics, never official tracking precision/score."""
    gain, labels, ranks = [], [], []
    selection = Counter()
    eligible_mothers, recovered_mothers = set(), set()
    for name, table in tables.items():
        ids, values, changed, y = (table[k] for k in ("ids", "scores", "changed", "labels"))
        v = values[:, col]
        known = (y == 1) | (y == -1)
        gain.extend(v[known])
        labels.extend(y[known])
        eligible_mothers.update((name, int(m)) for m in ids[y == 1, 0])
        best = p.best_positive(ids, v, changed)
        selected = best[v[best] > margin] if margin is not None else []
        for i in selected:
            label = int(y[i])
            selection[{1: "beneficial", -1: "harmful", 0: "mixed", 2: "unknown"}[label]] += 1
            if label == 1:
                recovered_mothers.add((name, int(ids[i, 0])))
        ranks.extend(table["fork_ranks"])
    gain, labels = np.asarray(gain, float), np.asarray(labels, int)
    pos, neg = labels == 1, labels == -1
    roc = None
    if pos.any() and neg.any():
        roc = float((rankdata(gain)[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
    decision = gain > margin if margin is not None else np.zeros(len(gain), bool)
    model_key = ("event013", "ranker012")[col]
    covered = [f for f in ranks if f["status"] == "covered"]
    tp, fp = int((decision & pos).sum()), int((decision & neg).sum())
    selected_total = sum(selection.values())
    return dict(confirmed_only=dict(beneficial=int(pos.sum()), harmful=int(neg.sum()), auroc=roc,
            tp=tp, fp=fp, fn=int(pos.sum())-tp, tn=int(neg.sum())-fp,
            recall=tp/int(pos.sum()) if pos.any() else None,
            fpr=fp/int(neg.sum()) if neg.any() else None),
        unique_best_selected={k: int(selection[k]) for k in ("beneficial", "harmful", "mixed", "unknown")},
        unknown_fraction_selected=selection["unknown"]/selected_total if selected_total else None,
        beneficial_mothers=len(eligible_mothers), recovered_beneficial_mothers=len(recovered_mothers),
        beneficial_mother_recall=len(recovered_mothers)/len(eligible_mothers) if eligible_mothers else None,
        gt_forks=len(ranks), covered_gt_forks=len(covered),
        unique_top1=sum(f["models"][model_key]["unique_top1"] for f in covered),
        new_unique_top1=sum(f["models"][model_key]["unique_top1"] and not f["already_present"] for f in covered),
        already_present_covered=sum(f["already_present"] for f in covered),
        margin=margin, unknowns_are_not_negatives=True, tracking_evaluated=False,
        scope="individual_hypotheses_before_overlap_resolution_not_official_metric")


def calibrate(tables, cfg):
    """One source-only order statistic, NOT a development-tuned margin sweep.

    Zero confirmed harms on this reused calibration is a construction property,
    not a population precision guarantee. Unknowns remain unverified. The result
    is diagnostic only and cannot authorize any edits.
    """
    harmful = [float(s) for t in tables.values() for s in t["scores"][t["labels"] == -1, 0]]
    trial = max(0., max(harmful)) if harmful else None
    summary = summarize_tables(tables, 0, trial)
    selected = summary["unique_best_selected"]
    reasons = []
    if len(harmful) < cfg["calibration_minimum_harmful"]:
        reasons.append("insufficient confirmed harmful alternatives")
    if selected["beneficial"] < cfg["calibration_minimum_beneficial_selected"]:
        reasons.append("no beneficial unique-best proposal above source-only margin")
    if summary["beneficial_mother_recall"] is None or summary["beneficial_mother_recall"] < cfg["calibration_minimum_beneficial_mother_recall"]:
        reasons.append("beneficial mother recall below source criterion")
    if selected["harmful"] or selected["mixed"]:
        reasons.append("confirmed harmful or mixed proposal selected")
    return dict(diagnostic_margin=trial if not reasons else None, trial_margin=trial,
        eligible=not reasons, reasons=reasons, summary=summary, rule=cfg["calibration_rule"],
        source_only=True, calibration_reused=True, precision_guarantee=False,
        authorizes_graph_edits=False)


def evaluate_prediction(folder, corpus, name, table):
    anchor, _, forks, ctx = source.open_evidence(folder.parent/config()["corpus_run"], name, corpus)
    # Prevent an incomplete or reordered score export from hiding alternatives.
    np.testing.assert_array_equal(table["ids"], p.hypothesis_ids(forks, ctx))
    if table["scores"].shape != (len(table["ids"]), 2) or not np.isfinite(table["scores"]).all():
        raise ValueError("Invalid paired prediction table")
    gt = Path(corpus["data_root"])/(name+".geff")
    before = source.tree_fingerprint(gt)
    mapping, children, parents, _ = source.match_annotation(anchor, gt, ctx["scale"])
    labels = annotation_labels(table["ids"], ctx, mapping, parents)
    np.testing.assert_array_equal(labels != NOOP, table["changed"])
    ranks = fork_ranks(table["ids"], table["scores"], table["changed"], mapping, children)
    if source.tree_fingerprint(gt) != before:
        raise ValueError("Evaluation annotations changed")
    return {**table, "labels": labels, "fork_ranks": ranks}, before


def safety():
    return dict(uses_gt_for_edits=False, graph_exported=False, tracking_evaluated=False,
        confirmation_evaluated=False, visible_evaluated=False, kaggle_submission_allowed=False,
        promoted=False, detector_full_fit=True, calibration_reused_from011=True, kaggle_score=None)


def discrimination_gate(by_group):
    reasons = []
    for group, row in by_group.items():
        new, old_metrics = row["event013_at_zero"], row["ranker012_at_zero"]
        a, b = new["confirmed_only"]["auroc"], old_metrics["confirmed_only"]["auroc"]
        if a is None or b is None or a+1e-12 < b:
            reasons.append(group+": confirmed-only AUROC regressed or undefined")
        if new["new_unique_top1"] < old_metrics["new_unique_top1"]:
            reasons.append(group+": new unique-top1 forks regressed")
        selected = row["event013_at_source_margin"]["unique_best_selected"]
        if selected["harmful"] or selected["mixed"]:
            reasons.append(group+": harmful/mixed best proposal at source margin")
    if not any(v["event013_at_zero"]["new_unique_top1"] > v["ranker012_at_zero"]["new_unique_top1"] for v in by_group.values()):
        reasons.append("no additional new unique-top1 fork across development")
    if not any(v["event013_at_source_margin"]["unique_best_selected"]["beneficial"] for v in by_group.values()):
        reasons.append("no beneficial best proposal at source-calibrated margin")
    return dict(passed=not reasons, reasons=reasons, authorizes_tracking=False)


def assay_job(args, run, folder):
    cfg = config()
    corpus = verify_sources(folder, code_manifest())
    data, summaries, hashes, models, comparators = {}, {}, {}, {}, {}
    for group in source.PROTOCOL["group_prefixes"]:
        data[group], summaries[group], gt = collect_training(folder, corpus, group, run)
        hashes.update(gt)
        comparators[group] = json.loads((folder/("ranker-"+group+".json")).read_text())
    def finish(result):
        for name, sha in hashes.items():
            if source.tree_fingerprint(Path(corpus["data_root"])/(name+".geff")) != sha:
                raise ValueError("Annotation source changed during013")
        verify_sources(folder, code_manifest())
        result.update(disposition=SCOPE, source_unchanged=True, gt_sha256=hashes, **safety())
        dump(run/"event-ranking.json", result)
        return {k: v for k, v in result.items() if k not in {"by_group", "gt_sha256"}} | {
            "event_ranking_sha256": digest(run/"event-ranking.json")}
    if not all(sufficient(s, cfg) for s in summaries.values()):
        return finish(dict(next_action="insufficient_supervision_no_submission", training_summary=summaries,
            model_trained=False, calibration_evaluated=False, development_evaluated=False,
            threshold_selection_attempted=False, threshold_selected=False))
    for group in data:
        ranker = fit_event_ranker(data[group], cfg)
        models[group] = dict(schema=cfg["schema"], scope=SCOPE, recipe=cfg, ranker=ranker,
            training_groups=[group], training_datasets=corpus["splits"][group]["fit"],
            selection_groups=[], selection_datasets=[], diagnostic_margin=None)
        dump(run/("fit-"+group+".json"), dict(model=models[group], training_summary=summaries[group],
            event013=preference_metrics(data[group], ranker),
            ranker012=preference_metrics(data[group], comparators[group]["ranker"]),
            metrics_scope="training_only_not_generalization"))
    del data

    def freeze(names, calibration):
        manifest = {}
        for name in names:
            target = corpus["datasets"][name]["group"]
            group = target if calibration else next(g for g in models if g != target)
            model_path = run/(("fit-" if calibration else "event-ranker-")+group+".json")
            saved_model = json.loads(model_path.read_text())
            if calibration:
                saved_model = saved_model["model"]
            if saved_model != models[group]:
                raise ValueError("Saved model differs from frozen prediction model")
            table = predict(folder, corpus, name, models[group], comparators[group], calibration)
            path = run/(name+".scores.npz")
            with path.open("xb") as handle:
                np.savez_compressed(handle, **table)
            manifest[name] = dict(path=path.name, sha256=digest(path), trained_on=group,
                model_path=model_path.name, model_sha256=digest(model_path),
                comparator_sha256=digest(folder/("ranker-"+group+".json")))
            print(json.dumps(dict(phase="event_score_prediction", dataset=name, hypotheses=len(table["ids"]))), flush=True)
        phase = "calibration" if calibration else "development"
        dump(run/(phase+"-prediction-manifest.json"), manifest)
        return manifest

    def annotate(manifest):
        tables = {}
        for name, info in manifest.items():
            path = run/info["path"]
            verify_file(path, info["sha256"])
            verify_file(run/info["model_path"], info["model_sha256"])
            with np.load(path, allow_pickle=False) as f:
                table = {k: f[k] for k in f.files}
            tables[name], hashes[name] = evaluate_prediction(folder, corpus, name, table)
            path = run/(name+".labels.npz")
            with path.open("xb") as handle:
                np.savez_compressed(handle, labels=tables[name]["labels"])
            dump(run/(name+".diagnostic.json"), dict(gt_sha256=hashes[name], labels_sha256=digest(path),
                score_sha256=info["sha256"], fork_ranks=tables[name]["fork_ranks"],
                label_codes=dict(beneficial=1, harmful=-1, mixed=0, unknown=2, reference_noop=3)))
        return tables

    # All10 source predictions exist before ANY calibration annotation is opened.
    calibration_names = [n for g in models for n in corpus["splits"][g]["calibration"]]
    calibration_manifest = freeze(calibration_names, True)
    tables = annotate(calibration_manifest)
    calibrations, specs = {}, {}
    for group, model in models.items():
        selected = calibrate({n: tables[n] for n in corpus["splits"][group]["calibration"]}, cfg)
        calibrations[group] = selected
        model.update(diagnostic_margin=selected["diagnostic_margin"], selection_groups=[group],
            selection_datasets=corpus["splits"][group]["calibration"])
        dump(run/("calibration-"+group+".json"), selected)
        path = run/("event-ranker-"+group+".json")
        dump(path, model)
        specs[group] = dict(path=path.name, sha256=digest(path), diagnostic_margin=model["diagnostic_margin"])
    dump(run/"models.json", specs)
    del tables
    # Even an abstaining source is diagnosed cross-prefix. No graph is edited,
    # target labels never select a margin, and confirmation remains inaccessible.
    development_manifest = freeze(corpus["development"], False)
    tables = annotate(development_manifest)
    by_group = {}
    for group in models:
        other = next(g for g in models if g != group)
        group_tables = {n: t for n, t in tables.items() if corpus["datasets"][n]["group"] == group}
        by_group[group] = dict(
            event013_at_zero=summarize_tables(group_tables, 0, 0.),
            ranker012_at_zero=summarize_tables(group_tables, 1, 0.),
            event013_at_source_margin=summarize_tables(group_tables, 0, models[other]["diagnostic_margin"]),
            ranker012_at_original_margin=summarize_tables(group_tables, 1, comparators[other]["margin"]))
    for spec in specs.values():
        verify_file(run/spec["path"], spec["sha256"])
    for manifest in (calibration_manifest, development_manifest):
        for info in manifest.values():
            verify_file(run/info["path"], info["sha256"])
            verify_file(run/info["model_path"], info["model_sha256"])
    gate = discrimination_gate(by_group)
    return finish(dict(next_action="designer_review_only" if gate["passed"] else "rejected_no_kaggle_submission",
        training_summary=summaries, models=specs, calibration=calibrations, by_group=by_group,
        discrimination_gate=gate, model_trained=True, calibration_evaluated=True, development_evaluated=True,
        threshold_selection_attempted=True, threshold_selected=any(m["diagnostic_margin"] is not None for m in models.values()),
        threshold_selection_scope="source_prefix_diagnostic_only", applied_transactions=0))
