"""015 stratified ranking: event-balanced preference learning with deduplicated supervision.

Branch A from docs/AUTONOMOUS_CONTINUATION.md and docs/CAMPAIGN_015_PREDECLARATION.md.
Integrates historical 012/013 preferences with Campaign 014 audited supervision
(including 44b6_a21120c2:12535 usable division fork, yielding 3 usable events in 3
volumes for 44b6 and 16 usable events in 11 volumes for 6bba).
Public annotations enter training and POST-prediction diagnostics only.
Models and calibration stay source-prefix-only. Unknowns are never treated as negatives.
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

ADDITIONS = {
    "core/stratified_ranking_015.py",
    "configs/event_ranker_015.json",
    "tests/test_stratified_ranking_015.py",
    "ops/audit_event_inputs_015.py",
}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "event_ranking_assay_only_no_submission"
UNKNOWN, MIXED, NOOP = 2, 0, 3


def config():
    return json.loads((ROOT / "configs/event_ranker_015.json").read_text())


def parent_manifest(current, archived):
    if hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest() != config()["source_release"]:
        raise ValueError("Unverified archived014 code manifest")
    if ({k: v for k, v in current.items() if k not in ADDITIONS | DRIVERS}
            != {k: v for k, v in archived.items() if k not in DRIVERS}):
        raise ValueError("Unreviewed modification to science/configuration001-014")
    return archived


def audit_inputs(runs, manifest=None):
    """Safe metadata/hash audit usable locally without opening GT or fitting."""
    runs, cfg = Path(runs), config()
    folder = runs / cfg["source_run"]
    for name, sha in cfg["source_artifacts"].items():
        verify_file(folder / name, sha)
    receipt = json.loads((folder / "receipt.json").read_text())
    archived = parent_manifest(code_manifest() if manifest is None else manifest, receipt["code_manifest"])
    if (receipt["status"] != "completed" or receipt["task"] != "audit-coverage"
            or receipt["code_hash"] != cfg["source_release"] or receipt["node"] != cfg["node"]):
        raise ValueError("Expected completed pinned014 source on A1555")
    report = json.loads((folder / "coverage.json").read_text())
    if report.get("graph_exported") is not False:
        raise ValueError("Unsupported014 outcome: graph was exported")

    # Verify corpus evidence
    corpus_dir = runs / cfg["corpus_run"]
    receipt011 = json.loads((corpus_dir / "receipt.json").read_text())
    corpus = json.loads((corpus_dir / "corpus.json").read_text())
    if len(corpus["datasets"]) != 53 or len(corpus["development"]) != 8:
        raise ValueError("Incomplete011 evidence")
    for name, info in corpus["datasets"].items():
        verify_file(corpus_dir / (name + ".evidence.npz"), info["sha256"])

    # Verify 012 ranking source
    rank_folder = runs / cfg["ranking_source_run"]
    for group in source.PROTOCOL["group_prefixes"]:
        prefs_p = rank_folder / ("preferences-" + group + ".npz")
        if not prefs_p.exists():
            raise ValueError(f"Missing 012 preferences for {group}")
        ranker_p = rank_folder / ("ranker-" + group + ".json")
        if not ranker_p.exists():
            raise ValueError(f"Missing 012 ranker for {group}")

    # Verify 013 event source
    events_folder = runs / cfg["events_source_run"]
    for group in source.PROTOCOL["group_prefixes"]:
        train_p = events_folder / ("training-" + group + ".npz")
        if not train_p.exists():
            raise ValueError(f"Missing 013 training for {group}")

    return (
        dict(
            passed=True,
            evidence_npz=53,
            source_release=cfg["source_release"],
            current_code_hash=code_hash(),
            gt_opened=False,
            historical005_chain_verified=False,
        ),
        corpus,
        receipt011["code_manifest"],
    )


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"]:
        raise ValueError(f"015 requires pinned {cfg['source_run']}; got {folder.name}")
    _, corpus, archived011 = audit_inputs(folder.parent, manifest)
    checked, _ = source.verify_corpus(folder.parent / cfg["corpus_run"], archived011)
    if checked != corpus:
        raise ValueError("Corpus changed during verification")
    return corpus


def training_summary(data):
    result = {}
    for kind in ("beneficial", "harmful", "event"):
        mask = data["strata"] == kind
        keys = set(data["keys"][mask])
        result[kind] = dict(
            comparisons=int(mask.sum()),
            events=len(keys),
            volumes=len({k.rsplit(":", 1)[0] for k in keys}),
        )
    result["unique_mothers_across_strata"] = len(set(data["keys"]))
    return result


def sufficient(summary, cfg):
    return (
        summary["beneficial"]["events"] >= cfg["minimum_beneficial_events"]
        and summary["harmful"]["events"] >= cfg["minimum_harmful_events"]
        and summary["event"]["events"] >= cfg["minimum_fork_events"]
        and summary["event"]["volumes"] >= cfg["minimum_fork_volumes"]
    )


def event_weights(keys, strata, cfg):
    keys, strata = np.asarray(keys), np.asarray(strata)
    mass = cfg["stratum_mass"]
    if (keys.shape != strata.shape or keys.ndim != 1 or not len(keys)
            or set(strata) != set(mass) or not np.isclose(sum(mass.values()), 1.0)
            or any(v <= 0 for v in mass.values())):
        raise ValueError("All three nonempty, event-balanced supervision strata required")
    weights = np.zeros(len(keys))
    for kind, amount in mass.items():
        mask = strata == kind
        counts = Counter(keys[mask])
        weights[mask] = [amount / (len(counts) * counts[k]) for k in keys[mask]]
    return weights


def assemble_training(folder, corpus, group, run=None):
    """Assemble deduplicated, event-balanced training preferences for group."""
    cfg = config()
    runs = folder.parent

    # 1. Base direct preferences from 012
    rank_folder = runs / cfg["ranking_source_run"]
    with np.load(rank_folder / ("preferences-" + group + ".npz"), allow_pickle=False) as f:
        x_012 = f["differences"]
        y_012 = f["signs"]
        k_012 = f["mother_keys"].tolist()

    # 2. Base event preferences from 013
    events_folder = runs / cfg["events_source_run"]
    with np.load(events_folder / ("training-" + group + ".npz"), allow_pickle=False) as f:
        ev_mask_013 = f["strata"] == "event"
        x_013_ev = f["x"][ev_mask_013]
        y_013_ev = f["y"][ev_mask_013]
        k_013_ev = f["keys"][ev_mask_013].tolist()

    if group == "44b6":
        # Load Campaign 014 new preferences
        with np.load(folder / "new-direct/preferences-44b6.npz", allow_pickle=False) as f:
            x_014 = f["differences"]
            y_014 = f["signs"]
            k_014 = f["mother_keys"]

        # Deduplicate: 44b6_a21120c2:12535 is placed into the 'event' stratum
        # with sign +1.0 (preferred true fork [12535, 12781, 12787] vs pruning).
        # In new-direct, difference was repair - reference with sign -1.0.
        # In event stratum, difference is good - other = -difference with sign +1.0.
        ev_mask_014 = k_014 == "44b6_a21120c2:12535"
        x_ev_014 = -x_014[ev_mask_014]
        y_ev_014 = np.ones(ev_mask_014.sum())
        k_ev_014 = k_014[ev_mask_014].tolist()

        # The remaining 215 unique comparisons remain in direct stratum
        x_dir_014 = x_014[~ev_mask_014]
        y_dir_014 = y_014[~ev_mask_014]
        k_dir_014 = k_014[~ev_mask_014].tolist()

        # Build combined arrays
        # Beneficial: y == 1 in 012 + y == 1 in remaining 014
        mask_ben_012 = np.asarray(y_012) == 1
        mask_ben_014 = np.asarray(y_dir_014) == 1
        x_ben = np.concatenate([x_012[mask_ben_012], x_dir_014[mask_ben_014]])
        y_ben = np.concatenate([np.asarray(y_012)[mask_ben_012], np.asarray(y_dir_014)[mask_ben_014]])
        k_ben = [k for k, m in zip(k_012, mask_ben_012) if m] + [k for k, m in zip(k_dir_014, mask_ben_014) if m]

        # Harmful: y == -1 in 012 + y == -1 in remaining 014
        mask_harm_012 = np.asarray(y_012) == -1
        mask_harm_014 = np.asarray(y_dir_014) == -1
        x_harm = np.concatenate([x_012[mask_harm_012], x_dir_014[mask_harm_014]])
        y_harm = np.concatenate([np.asarray(y_012)[mask_harm_012], np.asarray(y_dir_014)[mask_harm_014]])
        k_harm = [k for k, m in zip(k_012, mask_harm_012) if m] + [k for k, m in zip(k_dir_014, mask_harm_014) if m]

        # Event: 013 event + 014 usable event
        x_ev = np.concatenate([x_013_ev, x_ev_014])
        y_ev = np.concatenate([y_013_ev, y_ev_014])
        k_ev = k_013_ev + k_ev_014

        xs = np.concatenate([x_ben, x_harm, x_ev])
        ys = np.concatenate([y_ben, y_harm, y_ev])
        keys = k_ben + k_harm + k_ev
        strata = ["beneficial"] * len(y_ben) + ["harmful"] * len(y_harm) + ["event"] * len(y_ev)
    else:
        # 6bba already has full 3-strata representation from 012 and 013
        mask_ben_012 = np.asarray(y_012) == 1
        mask_harm_012 = np.asarray(y_012) == -1
        x_ben = x_012[mask_ben_012]
        y_ben = np.asarray(y_012)[mask_ben_012]
        k_ben = [k for k, m in zip(k_012, mask_ben_012) if m]

        x_harm = x_012[mask_harm_012]
        y_harm = np.asarray(y_012)[mask_harm_012]
        k_harm = [k for k, m in zip(k_012, mask_harm_012) if m]

        x_ev = x_013_ev
        y_ev = y_013_ev
        k_ev = k_013_ev

        xs = np.concatenate([x_ben, x_harm, x_ev])
        ys = np.concatenate([y_ben, y_harm, y_ev])
        keys = k_ben + k_harm + k_ev
        strata = ["beneficial"] * len(y_ben) + ["harmful"] * len(y_harm) + ["event"] * len(y_ev)

    data = dict(
        x=xs,
        y=np.asarray(ys, float),
        keys=np.asarray(keys, str),
        strata=np.asarray(strata, str),
    )
    summary = training_summary(data)
    if run is not None:
        run = Path(run)
        path = run / ("training-" + group + ".npz")
        with path.open("xb") as handle:
            np.savez_compressed(handle, **data)
        dump(
            run / ("training-" + group + ".json"),
            dict(
                summary=summary,
                tensor_sha256=digest(path),
                source_coverage=folder.name,
                deduplicated=True,
            ),
        )
    return data, summary


collect_training = assemble_training


def fit_event_ranker(data, cfg, variant="variant1"):
    """Fit regularized linear ranking potential.

    Variant 1: Standard L2 on event-balanced weights.
    Variant 2: Stratified L2 (higher event weight).
    """
    x, y, strata, keys = data["x"], data["y"], data["strata"], data["keys"]
    if (x.shape != (len(y), len(r.NAMES)) or len(strata) != len(y)
            or not np.isfinite(x).all() or not np.isin(y, [-1, 1]).all()
            or not (y[strata == "harmful"] == -1).all()
            or not (y[strata != "harmful"] == 1).all()):
        raise ValueError("Invalid confirmed training preferences")
    weights = event_weights(keys, strata, cfg)
    rms = np.sqrt(np.sum(weights[:, None] * x * x, axis=0))
    active = rms >= cfg["std_floor"]
    scale = np.where(active, rms, 1.0)
    signed = x / scale * active * y[:, None]

    l2 = cfg["l2"] if variant == "variant1" else cfg["l2"] * 1.5
    fit = minimize(
        old.rank_objective,
        np.zeros(x.shape[1]),
        args=(signed, weights, l2),
        jac=True,
        method="L-BFGS-B",
        options={k: cfg[k] for k in ("maxiter", "ftol", "gtol")},
    )
    loss, grad = old.rank_objective(fit.x, signed, weights, l2)
    norm = float(np.max(np.abs(grad)))
    if not fit.success or not np.isfinite(loss) or not np.isfinite(norm) or norm > cfg["max_gradient_inf"]:
        raise ValueError(f"015 optimization failed for {variant}: {fit.message}; gradient_inf={norm}")
    return dict(
        feature_names=list(r.NAMES),
        weights=fit.x.tolist(),
        intercept=0.0,
        standardizer=dict(mean=[0.0] * x.shape[1], scale=scale.tolist(), active=active.tolist()),
        fit=dict(
            objective=loss,
            gradient_inf=norm,
            iterations=int(fit.nit),
            variant=variant,
            l2=l2,
            stratum_mass={k: float(weights[strata == k].sum()) for k in cfg["stratum_mass"]},
        ),
        calibrated_probability=False,
    )


fit_stratified_ranker = fit_event_ranker


def preference_metrics(data, model):
    gain = r.scores(model, data["x"])
    result = {}
    for kind in config()["stratum_mass"]:
        take = data["strata"] == kind
        satisfied = gain[take] * data["y"][take] > 0
        keys = data["keys"][take]
        per_event = [float(satisfied[keys == k].mean()) for k in sorted(set(keys))]
        result[kind] = dict(
            comparisons=int(take.sum()),
            satisfied=int(satisfied.sum()),
            fraction=float(satisfied.mean()) if len(satisfied) else None,
            event_macro_fraction=float(np.mean(per_event)) if per_event else None,
        )
    return result


def check_model(model, corpus, name, calibration=False):
    trained = model.get("training_groups", [])
    if (model.get("schema") != "event_ranker_015" or model.get("scope") != SCOPE
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
    _, _, forks, ctx = source.open_evidence(folder.parent / config()["corpus_run"], name, corpus)
    ids, values, changed = p.score_universe(
        forks, ctx, [model["ranker"], comparator["ranker"]], config()["feature_batch_size"]
    )
    return dict(ids=ids, scores=values, changed=changed)


def edge_truth(mapping, parents):
    inverse = {int(v): int(k) for k, v in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Non-injective annotation matching")

    def label(edge):
        a, b = (inverse.get(int(i)) for i in edge)
        return None if a is None or b not in parents else int(parents[b] == a)

    return label


def contrast_label(good, other, truth):
    ga, gr, oa, ore = [set(map(tuple, es)) for es in
                       (good["add"], good["remove"], other["add"], other["remove"])]
    add, remove = (ga - oa) | (ore - gr), (gr - ore) | (oa - ga)
    if not add and not remove:
        return NOOP
    labels = [truth(e) for e in sorted(add) + sorted(remove)]
    if any(v is None for v in labels):
        return UNKNOWN
    delta = [2 * v - 1 for v in labels[:len(add)]] + [1 - 2 * v for v in labels[len(add):]]
    return 1 if all(v == 1 for v in delta) else -1 if all(v == -1 for v in delta) else MIXED


def annotation_labels(ids, ctx, mapping, parents):
    truth = edge_truth(mapping, parents)
    return np.asarray([contrast_label(p.transaction(i, ctx), p.reference(i[0]), truth)
                       for i in ids], np.int8)


def fork_ranks(ids, values, changed, mapping, children):
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
                for col, key in enumerate(("event015", "ranker012")):
                    other = values[choices[choices != i], col]
                    if changed[i] and not (~changed[choices]).any():
                        other = np.append(other, 0.0)
                    worst = 1 + int((other >= values[i, col] - 1e-10).sum())
                    row["models"][key] = dict(
                        gain=float(values[i, col]), rank_worst=worst, unique_top1=worst == 1
                    )
        rows.append(row)
    return rows


def summarize_tables(tables, col, margin):
    gain, labels, ranks = [], [], []
    selection = Counter()
    eligible_mothers, recovered_mothers = set(), set()
    for name, table in tables.items():
        ids, values, changed, y = table["ids"], table["scores"], table["changed"], table["labels"]
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
        roc = float((rankdata(gain)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
    decision = gain > margin if margin is not None else np.zeros(len(gain), bool)
    model_key = ("event015", "ranker012")[col]
    covered = [f for f in ranks if f["status"] == "covered"]
    tp, fp = int((decision & pos).sum()), int((decision & neg).sum())
    selected_total = sum(selection.values())
    return dict(
        confirmed_only=dict(
            beneficial=int(pos.sum()),
            harmful=int(neg.sum()),
            auroc=roc,
            tp=tp,
            fp=fp,
            fn=int(pos.sum()) - tp,
            tn=int(neg.sum()) - fp,
            recall=tp / int(pos.sum()) if pos.any() else None,
            fpr=fp / int(neg.sum()) if neg.any() else None,
        ),
        unique_best_selected={k: int(selection[k]) for k in ("beneficial", "harmful", "mixed", "unknown")},
        unknown_fraction_selected=selection["unknown"] / selected_total if selected_total else None,
        beneficial_mothers=len(eligible_mothers),
        recovered_beneficial_mothers=len(recovered_mothers),
        beneficial_mother_recall=len(recovered_mothers) / len(eligible_mothers) if eligible_mothers else None,
        gt_forks=len(ranks),
        covered_gt_forks=len(covered),
        unique_top1=sum(f["models"][model_key]["unique_top1"] for f in covered),
        new_unique_top1=sum(f["models"][model_key]["unique_top1"] and not f["already_present"] for f in covered),
        already_present_covered=sum(f["already_present"] for f in covered),
        margin=margin,
        unknowns_are_not_negatives=True,
        tracking_evaluated=False,
        scope="individual_hypotheses_before_overlap_resolution_not_official_metric",
    )


def calibrate(tables, cfg):
    harmful = [float(s) for t in tables.values() for s in t["scores"][t["labels"] == -1, 0]]
    trial = max(0.0, max(harmful)) if harmful else None
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
    return dict(
        diagnostic_margin=trial if not reasons else None,
        trial_margin=trial,
        eligible=not reasons,
        reasons=reasons,
        summary=summary,
        rule=cfg["calibration_rule"],
        source_only=True,
        calibration_reused=True,
        precision_guarantee=False,
        authorizes_graph_edits=False,
    )


def evaluate_prediction(folder, corpus, name, table):
    anchor, _, forks, ctx = source.open_evidence(folder.parent / config()["corpus_run"], name, corpus)
    np.testing.assert_array_equal(table["ids"], p.hypothesis_ids(forks, ctx))
    if table["scores"].shape != (len(table["ids"]), 2) or not np.isfinite(table["scores"]).all():
        raise ValueError("Invalid paired prediction table")
    gt = Path(corpus["data_root"]) / (name + ".geff")
    before = source.tree_fingerprint(gt)
    mapping, children, parents, _ = source.match_annotation(anchor, gt, ctx["scale"])
    labels = annotation_labels(table["ids"], ctx, mapping, parents)
    np.testing.assert_array_equal(labels != NOOP, table["changed"])
    ranks = fork_ranks(table["ids"], table["scores"], table["changed"], mapping, children)
    if source.tree_fingerprint(gt) != before:
        raise ValueError("Evaluation annotations changed")
    return {**table, "labels": labels, "fork_ranks": ranks}, before


def discrimination_gate(by_group):
    reasons = []
    for group, row in by_group.items():
        new, old_metrics = row["event015_at_zero"], row["ranker012_at_zero"]
        a, b = new["confirmed_only"]["auroc"], old_metrics["confirmed_only"]["auroc"]
        if a is None or b is None or a + 1e-12 < b:
            reasons.append(group + ": confirmed-only AUROC regressed or undefined")
        if new["new_unique_top1"] < old_metrics["new_unique_top1"]:
            reasons.append(group + ": new unique-top1 forks regressed")
        selected = row["event015_at_source_margin"]["unique_best_selected"]
        if selected["harmful"] or selected["mixed"]:
            reasons.append(group + ": harmful/mixed best proposal at source margin")
    if not any(v["event015_at_zero"]["new_unique_top1"] > v["ranker012_at_zero"]["new_unique_top1"] for v in by_group.values()):
        reasons.append("no additional new unique-top1 fork across development")
    if not any(v["event015_at_source_margin"]["unique_best_selected"]["beneficial"] for v in by_group.values()):
        reasons.append("no beneficial best proposal at source-calibrated margin")
    return dict(passed=not reasons, reasons=reasons, authorizes_tracking=False)


def safety():
    return dict(
        uses_gt_for_edits=False,
        graph_exported=False,
        tracking_evaluated=False,
        confirmation_evaluated=False,
        visible_evaluated=False,
        kaggle_submission_allowed=False,
        promoted=False,
        detector_full_fit=True,
        calibration_reused_from011=True,
        kaggle_score=None,
    )


def assay_job(args, run, folder):
    cfg = config()
    corpus = verify_sources(folder, code_manifest())
    data, summaries, hashes, models, comparators = {}, {}, {}, {}, {}
    rank_folder = folder.parent / cfg["ranking_source_run"]
    for group in source.PROTOCOL["group_prefixes"]:
        data[group], summaries[group] = assemble_training(folder, corpus, group, run)
        comparators[group] = json.loads((rank_folder / ("ranker-" + group + ".json")).read_text())

    def finish(result):
        for name, sha in hashes.items():
            if source.tree_fingerprint(Path(corpus["data_root"]) / (name + ".geff")) != sha:
                raise ValueError("Annotation source changed during015")
        verify_sources(folder, code_manifest())
        result.update(disposition=SCOPE, source_unchanged=True, gt_sha256=hashes, **safety())
        dump(run / "event-ranking.json", result)
        return {k: v for k, v in result.items() if k not in {"by_group", "gt_sha256"}} | {
            "event_ranking_sha256": digest(run / "event-ranking.json")
        }

    if not all(sufficient(s, cfg) for s in summaries.values()):
        return finish(
            dict(
                next_action="insufficient_supervision_no_submission",
                training_summary=summaries,
                model_trained=False,
                calibration_evaluated=False,
                development_evaluated=False,
                threshold_selection_attempted=False,
                threshold_selected=False,
            )
        )

    for group in data:
        ranker = fit_event_ranker(data[group], cfg)
        models[group] = dict(
            schema=cfg["schema"],
            scope=SCOPE,
            recipe=cfg,
            ranker=ranker,
            training_groups=[group],
            training_datasets=corpus["splits"][group]["fit"],
            selection_groups=[],
            selection_datasets=[],
            diagnostic_margin=None,
        )
        dump(
            run / ("fit-" + group + ".json"),
            dict(
                model=models[group],
                training_summary=summaries[group],
                event015=preference_metrics(data[group], ranker),
                ranker012=preference_metrics(data[group], comparators[group]["ranker"]),
                metrics_scope="training_only_not_generalization",
            ),
        )
    del data

    def freeze(names, calibration):
        manifest = {}
        for name in names:
            target = corpus["datasets"][name]["group"]
            group = target if calibration else next(g for g in models if g != target)
            model_path = run / (("fit-" if calibration else "event-ranker-") + group + ".json")
            saved_model = json.loads(model_path.read_text())
            if calibration:
                saved_model = saved_model["model"]
            if saved_model != models[group]:
                raise ValueError("Saved model differs from frozen prediction model")
            table = predict(folder, corpus, name, models[group], comparators[group], calibration)
            path = run / (name + ".scores.npz")
            with path.open("xb") as handle:
                np.savez_compressed(handle, **table)
            manifest[name] = dict(
                path=path.name,
                sha256=digest(path),
                trained_on=group,
                model_path=model_path.name,
                model_sha256=digest(model_path),
                comparator_sha256=digest(rank_folder / ("ranker-" + group + ".json")),
            )
            print(
                json.dumps(
                    dict(phase="event_score_prediction", dataset=name, hypotheses=len(table["ids"]))
                ),
                flush=True,
            )
        phase = "calibration" if calibration else "development"
        dump(run / (phase + "-prediction-manifest.json"), manifest)
        return manifest

    def annotate(manifest):
        tables = {}
        for name, info in manifest.items():
            path = run / info["path"]
            verify_file(path, info["sha256"])
            verify_file(run / info["model_path"], info["model_sha256"])
            with np.load(path, allow_pickle=False) as f:
                table = {k: f[k] for k in f.files}
            tables[name], hashes[name] = evaluate_prediction(folder, corpus, name, table)
            path = run / (name + ".labels.npz")
            with path.open("xb") as handle:
                np.savez_compressed(handle, labels=tables[name]["labels"])
            dump(
                run / (name + ".diagnostic.json"),
                dict(
                    gt_sha256=hashes[name],
                    labels_sha256=digest(path),
                    score_sha256=info["sha256"],
                    fork_ranks=tables[name]["fork_ranks"],
                    label_codes=dict(beneficial=1, harmful=-1, mixed=0, unknown=2, reference_noop=3),
                ),
            )
        return tables

    # Calibration phase
    calibration_names = [n for g in models for n in corpus["splits"][g]["calibration"]]
    calibration_manifest = freeze(calibration_names, True)
    tables = annotate(calibration_manifest)
    calibrations, specs = {}, {}
    for group, model in models.items():
        selected = calibrate({n: tables[n] for n in corpus["splits"][group]["calibration"]}, cfg)
        calibrations[group] = selected
        model.update(
            diagnostic_margin=selected["diagnostic_margin"],
            selection_groups=[group],
            selection_datasets=corpus["splits"][group]["calibration"],
        )
        dump(run / ("calibration-" + group + ".json"), selected)
        path = run / ("event-ranker-" + group + ".json")
        dump(path, model)
        specs[group] = dict(path=path.name, sha256=digest(path), diagnostic_margin=model["diagnostic_margin"])
    dump(run / "models.json", specs)
    del tables

    # Development phase
    development_manifest = freeze(corpus["development"], False)
    tables = annotate(development_manifest)
    by_group = {}
    for group in models:
        other = next(g for g in models if g != group)
        group_tables = {n: t for n, t in tables.items() if corpus["datasets"][n]["group"] == group}
        by_group[group] = dict(
            event015_at_zero=summarize_tables(group_tables, 0, 0.0),
            ranker012_at_zero=summarize_tables(group_tables, 1, 0.0),
            event015_at_source_margin=summarize_tables(group_tables, 0, models[other]["diagnostic_margin"]),
            ranker012_at_original_margin=summarize_tables(group_tables, 1, comparators[other]["margin"]),
        )
    for spec in specs.values():
        verify_file(run / spec["path"], spec["sha256"])
    for manifest in (calibration_manifest, development_manifest):
        for info in manifest.values():
            verify_file(run / info["path"], info["sha256"])
            verify_file(run / info["model_path"], info["model_sha256"])
    gate = discrimination_gate(by_group)
    return finish(
        dict(
            next_action="designer_review_only" if gate["passed"] else "rejected_no_kaggle_submission",
            training_summary=summaries,
            models=specs,
            calibration=calibrations,
            by_group=by_group,
            discrimination_gate=gate,
            model_trained=True,
            calibration_evaluated=True,
            development_evaluated=True,
            threshold_selection_attempted=True,
            threshold_selected=any(m["diagnostic_margin"] is not None for m in models.values()),
            threshold_selection_scope="source_prefix_diagnostic_only",
            applied_transactions=0,
        )
    )
