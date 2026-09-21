"""012 source-only learning of preferences between whole repair transactions.

Only this workflow reads public annotations. It reuses immutable011 measured
evidence; it never calls a detector, reads confirmation or exports a submission.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core import mitosis_selector as previous
from core.nextgen import repair_ranker as policy
from core.nextgen.association import apply_transactions
from core.nextgen.contracts import Lineage, verify_file

ADDITIONS = {"core/nextgen/repair_ranker.py", "core/repair_ranking.py",
    "configs/repair_ranker.json", "tests/test_repair_ranking.py", "ops/audit_selector_inputs.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
SCOPE = "repair_ranking_development_only_no_submission"


def config():
    return json.loads((ROOT/"configs/repair_ranker.json").read_text())


def parent_manifest(manifest, archived):
    expected = config()["source_release"]
    if hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError("Archived011 code manifest does not match frozen release")
    if ({p: h for p, h in manifest.items() if p not in ADDITIONS | DRIVERS}
            != {p: h for p, h in archived.items() if p not in DRIVERS}):
        raise ValueError("Unreviewed change to pre012 science/configuration")
    return archived


def audit_inputs(runs):
    """Read-only local/remote011 retrieval audit. Never opens image or GT paths.

    This audits011 input bytes, not the availability of the complete005 chain.
    Remote verify_sources additionally verifies that historical scientific chain.
    """
    cfg, runs = config(), Path(runs)
    for relative, sha in cfg["source_artifacts"].items():
        verify_file(runs/relative, sha)
    corpus_dir, result_dir = runs/cfg["source_run"], runs/"b0970-selector-011"
    receipt = json.loads((corpus_dir/"receipt.json").read_text())
    archived = parent_manifest(code_manifest(), receipt["code_manifest"])
    corpus = json.loads((corpus_dir/"corpus.json").read_text())
    report = json.loads((result_dir/"selector.json").read_text())
    if (len(corpus["datasets"]) != 53 or len(corpus["development"]) != 8
            or report["calibration_margins"] != {"44b6": None, "6bba": None}
            or report["confirmation_evaluated"] is not False
            or report["kaggle_submission_allowed"] is not False):
        raise ValueError("Unsupported011 source outcome")
    for name, row in corpus["datasets"].items():
        verify_file(corpus_dir/(name+".evidence.npz"), row["sha256"])
    for row in json.loads((result_dir/"models.json").read_text()).values():
        verify_file(result_dir/row["path"], row["sha256"])
    for name, sha in json.loads((result_dir/"calibration-prediction-manifest.json").read_text()).items():
        verify_file(result_dir/(name+".calibration-proposals.json"), sha)
    predictions = json.loads((result_dir/"development-prediction-manifest.json").read_text())
    if set(predictions) != set(corpus["development"]):
        raise ValueError("Incomplete011 development predictions")
    for name, info in predictions.items():
        path = result_dir/(name+".prediction.npz")
        verify_file(path, info["sha256"])
        #011 selected abstention: verify arrays, not merely rounded metric parity.
        with np.load(path, allow_pickle=False) as pred, np.load(
                corpus_dir/(name+".evidence.npz"), allow_pickle=False) as evidence:
            np.testing.assert_array_equal(pred["coords"], evidence["coords"])
            np.testing.assert_array_equal(pred["edges"], evidence["reference_edges"])
    return dict(passed=True, evidence_npz=53, prediction_npz=8, abstention_arrays_equal=True,
        source_release=cfg["source_release"], current_code_hash=code_hash(),
        gt_opened=False, historical005_chain_verified=False), archived


def verify_sources(folder, manifest):
    if folder.name != config()["source_run"]:
        raise ValueError("012 requires the exact existing011 corpus; do not rerun preparation")
    _, archived = audit_inputs(folder.parent)
    parent_manifest(manifest, archived)
    # Pass the VERIFIED historical manifest, not a fabricated current-release receipt.
    return previous.verify_corpus(folder, archived)


def preference(tx, mapping, parents):
    """Strict partial-label preference, NOT an official metric or oracle edit.

    +1: every changed edge improves its annotated status. -1: every change
    worsens it. None: at least one unknown endpoint/parent. 0: mixed tradeoff.
    Unknowns and mixed trades are never silently treated as negatives.
    """
    added, removed = tx["add"], tx["remove"]
    labels = previous.edge_labels(added+removed, mapping, parents)
    if not len(labels):
        return 0
    if (labels < 0).any():
        return None
    changes = np.concatenate([2*labels[:len(added)]-1, 1-2*labels[len(added):]])
    if (changes == 1).all():
        return 1
    if (changes == -1).all():
        return -1
    return 0


def collect_group(folder, corpus, group, out):
    xs, ys, keys, hypothesis_ids, evidence = [], [], [], [], {}
    for name in corpus["splits"][group]["fit"]:
        if corpus["datasets"][name]["group"] != group:
            raise ValueError("Target-prefix training leakage")
        anchor, _, forks, ctx = previous.open_evidence(folder, name, corpus)
        gt = Path(corpus["data_root"])/(name+".geff")
        before = previous.tree_fingerprint(gt)
        mapping, _, parents, _ = previous.match_annotation(anchor, gt, ctx["scale"])
        # Necessary condition for FULLY KNOWN new fork edges. Training only;
        # inference still enumerates the entire53-file frozen proposal universe.
        mothers = np.asarray(list(mapping.values()), np.int64)
        daughters = np.asarray([v for g, v in mapping.items() if g in parents], np.int64)
        eligible = np.isin(forks[:, 0], mothers) & np.isin(forks[:, 1], daughters) & np.isin(forks[:, 2], daughters)
        known_forks = forks[eligible]
        fork_labels = previous.label_pairs(known_forks, mapping, parents)
        counts = Counter()
        for batch in policy.batches(policy.transactions(known_forks, ctx), config()["feature_batch_size"]):
            chosen, signs, group_keys = [], [], []
            for tx in batch:
                label = preference(tx, mapping, parents)
                counts[{None: "unknown", 0: "mixed", 1: "beneficial", -1: "harmful"}[label]] += 1
                if label in (-1, 1):
                    chosen.append(tx)
                    signs.append(label)
                    group_keys.append(name+":"+str(tx["mother"]))
                    hypothesis_ids.append([tx["mother"], *tx["daughters"], *([-1] if len(tx["daughters"]) == 1 else [])])
            if chosen:
                xs.append(policy.features(chosen, ctx))
                ys.extend(signs)
                keys.extend(group_keys)
        if previous.tree_fingerprint(gt) != before:
            raise ValueError("Training annotations changed")
        evidence[name] = dict(gt_sha256=before, fork_pool=len(forks),
            fork_training_screened_out=int((~eligible).sum()),
            confirmed_true_forks=int((fork_labels == 1).sum()), preferences=dict(counts))
        print(json.dumps(dict(phase="ranker_preferences", dataset=name, **evidence[name])), flush=True)
    x = np.concatenate(xs) if xs else np.empty((0, len(policy.NAMES)))
    y = np.asarray(ys, float)
    stats = dict(preferences=len(y), mothers=len(set(keys)), beneficial=int((y == 1).sum()),
        harmful=int((y == -1).sum()), confirmed_true_forks=sum(e["confirmed_true_forks"] for e in evidence.values()))
    path = out/("preferences-"+group+".npz")
    with path.open("xb") as handle:
        np.savez_compressed(handle, differences=x, signs=y, mother_keys=np.asarray(keys, dtype=str),
            hypothesis_ids=np.asarray(hypothesis_ids, np.int64).reshape(-1, 3))
    dump(out/("preferences-"+group+".json"), dict(summary=stats, datasets=evidence,
        interpretation="reference-versus-repair preferences; not independent biological events",
        mixed_and_unknown_excluded=True, training_only=True,
        tensor_path=path.name, tensor_sha256=digest(path)))
    return x, y, keys, evidence, stats


def mother_weights(keys):
    counts = Counter(keys)
    if not counts:
        raise ValueError("Empty preference corpus")
    return np.asarray([1./(len(counts)*counts[k]) for k in keys])


def rank_objective(w, signed_x, weights, l2):
    margin = signed_x @ w
    loss = np.sum(weights*np.logaddexp(0., -margin)) + .5*l2*np.dot(w, w)
    grad = -(signed_x.T @ (weights*expit(-margin))) + l2*w
    return float(loss), grad


def fit_ranker(x, y, keys, cfg):
    if (x.shape != (len(y), len(policy.NAMES)) or len(keys) != len(y) or not len(y)
            or not np.isfinite(x).all() or not np.isin(y, [-1, 1]).all()):
        raise ValueError("Finite confirmed preference differences required")
    base = mother_weights(keys)
    # Scale differences about ZERO. Centering/free intercept would destroy the
    # identity Phi(reference)-Phi(reference)=0 and reversal antisymmetry.
    rms = np.sqrt(np.sum(base[:, None]*x*x, axis=0))
    active = rms >= cfg["std_floor"]
    scale = np.where(active, rms, 1.)
    z = x/scale*active
    signed = z*y[:, None]
    audits, learned = [], np.zeros(x.shape[1])
    weight, hard = base, []
    for round_id in range(2):
        fit = minimize(rank_objective, learned, args=(signed, weight, cfg["l2"]), jac=True,
            method="L-BFGS-B", options={k: cfg[k] for k in ("maxiter", "ftol", "gtol")})
        loss, gradient = rank_objective(fit.x, signed, weight, cfg["l2"])
        norm = float(np.max(np.abs(gradient)))
        if not fit.success or not np.isfinite(loss) or norm > cfg["max_gradient_inf"]:
            raise ValueError(f"Ranker optimization failed: {fit.message}; gradient_inf={norm}")
        learned = fit.x
        margins = signed @ learned
        audits.append(dict(round=round_id+1, objective=loss, gradient_inf=norm,
            iterations=int(fit.nit), training_preference_accuracy=float((margins > 0).mean()),
            training_min_margin=float(margins.min()), hard_indices=hard))
        if round_id == 0:
            # Fixed ONE hard-preference mining round, source fit only. Never
            # mine calibration, target predictions, unknowns or oracle010 edits.
            worst = {}
            for i, key in enumerate(keys):
                if key not in worst or margins[i] < margins[worst[key]]:
                    worst[key] = i
            hard = sorted(worst.values())
            alpha = cfg["hard_preference_fraction"]
            weight = (1.-alpha)*base
            weight[hard] += alpha/len(hard)
    return dict(feature_names=list(policy.NAMES), weights=learned.tolist(), intercept=0.,
        standardizer=dict(mean=[0.]*x.shape[1], scale=scale.tolist(), active=active.tolist()),
        fit_rounds=audits, calibrated_probability=False, same_corpus_reused_not_independent_events=True)


def sufficient(stats, cfg):
    return (stats["preferences"] >= cfg["minimum_preferences"]
        and stats["mothers"] >= cfg["minimum_mothers"]
        and stats["confirmed_true_forks"] >= cfg["minimum_annotated_forks"])


def check_model(model, corpus, name, calibration=False):
    group = corpus["datasets"][name]["group"]
    trained = model.get("training_groups", [])
    if (model.get("schema") != "repair_ranker_012" or model.get("scope") != SCOPE
            or model.get("recipe") != config() or len(trained) != 1
            or trained[0] not in previous.PROTOCOL["group_prefixes"]
            or model.get("training_datasets") != corpus["splits"][trained[0]]["fit"]
            or model.get("margin") not in [None, *config()["margin_grid"]]):
        raise ValueError("Invalid ranker model/provenance")
    source = trained[0]
    if calibration:
        if (group != source or name not in corpus["splits"][source]["calibration"]
                or model["selection_groups"] or model["selection_datasets"] or model.get("margin") is not None):
            raise ValueError("Only unselected source-prefix calibration allowed")
    elif (group == source or name not in corpus["development"]
            or model["selection_groups"] != [source]
            or model["selection_datasets"] != corpus["splits"][source]["calibration"]):
        raise ValueError("Target prefix entered fitting/calibration or panel is not development")


def predict(folder, corpus, name, model, calibration=False):
    check_model(model, corpus, name, calibration)
    anchor, _, forks, ctx = previous.open_evidence(folder, name, corpus)
    rows, audit = policy.proposals(forks, ctx, model["ranker"], config())
    return anchor, rows, audit


def safety(tracking=False):
    return dict(confirmation_evaluated=False, visible_evaluated=False, uses_gt_for_edits=False,
        tracking_evaluated=tracking, kaggle_submission_allowed=False, promoted=False,
        detector_full_fit=True, calibration_reused_from011=True,
        scope="public_paired_diagnostic_not_independent_holdout_not_kaggle_score")


def evaluate_job(args, run, folder):
    cfg = config()
    corpus, archived = verify_sources(folder, code_manifest())
    data, models, gt_hashes = {}, {}, {}
    for group in previous.PROTOCOL["group_prefixes"]:
        data[group] = collect_group(folder, corpus, group, run)
        gt_hashes.update({n: e["gt_sha256"] for n, e in data[group][3].items()})
    ready = all(sufficient(d[4], cfg) for d in data.values())
    if not ready:
        for name, sha in gt_hashes.items():
            if previous.tree_fingerprint(Path(corpus["data_root"])/(name+".geff")) != sha:
                raise ValueError("Training annotations changed")
        verify_sources(folder, code_manifest())
        result = dict(disposition=SCOPE, next_action="insufficient_supervision_no_submission",
            training_summary={g: d[4] for g, d in data.items()}, model_trained=False,
            threshold_selected=False, calibration_evaluated=False, source_unchanged=True, **safety())
        dump(run/"ranking.json", result)
        return {**result, "ranking_sha256": digest(run/"ranking.json")}
    for group, (x, y, keys, evidence, stats) in data.items():
        ranker = fit_ranker(x, y, keys, cfg)
        models[group] = dict(schema="repair_ranker_012", scope=SCOPE, recipe=cfg, ranker=ranker,
            training_groups=[group], training_datasets=corpus["splits"][group]["fit"],
            selection_groups=[], selection_datasets=[], margin=None)
        dump(run/("fit-"+group+".json"), dict(model=models[group], training_summary=stats))
    del data
    # Both fits, then ALL ten calibration proposals, then any calibration metrics.
    frozen = {}
    for group in models:
        for name in corpus["splits"][group]["calibration"]:
            _, rows, audit = predict(folder, corpus, name, models[group], calibration=True)
            path = run/(name+".calibration-proposals.json")
            dump(path, dict(proposals=rows, audit=audit, trained_on=group))
            frozen[name] = digest(path)
            print(json.dumps(dict(phase="ranker_calibration_prediction", dataset=name, **audit)), flush=True)
    dump(run/"calibration-prediction-manifest.json", frozen)
    for group in models:
        refs, candidates, edits = {}, {}, {}
        margin = cfg["margin_grid"][0]
        for name in corpus["splits"][group]["calibration"]:
            path = run/(name+".calibration-proposals.json")
            verify_file(path, frozen[name])
            rows = json.loads(path.read_text())["proposals"]
            anchor, _, _, _ = previous.open_evidence(folder, name, corpus)
            graph, edits[name] = apply_transactions(anchor, rows, margin, cfg)
            row = previous.resolve_dataset_row(name, corpus["data_root"], split="calibration", gt=True)
            gt_hashes[name] = previous.tree_fingerprint(Path(row["gt"]))
            refs[name] = previous.metric_row(anchor, row)
            candidates[name] = previous.metric_row(graph, row)
        trial = dict(evaluation=previous.evaluate_panel(refs, candidates), edits=edits,
            budget_rejected=any(e["status"] == "rejected_edit_budget" for e in edits.values()))
        chosen = previous.select_margin({margin: trial}, cfg)
        models[group].update(margin=chosen, selection_groups=[group],
            selection_datasets=corpus["splits"][group]["calibration"])
        dump(run/("calibration-"+group+".json"), dict(selected_margin=chosen, trials={margin: trial},
            reused_from011=True, independent_holdout=False))
    specs = {}
    for group, model in models.items():
        path = run/("ranker-"+group+".json")
        dump(path, model)
        specs[group] = dict(path=path.name, sha256=digest(path), margin=model["margin"])
    dump(run/"models.json", specs)
    predicted, refs, predictions, audits = {}, {}, {}, {}
    for name in corpus["development"]:
        source = next(g for g in models if g != corpus["datasets"][name]["group"])
        verify_file(run/specs[source]["path"], specs[source]["sha256"])
        model = json.loads((run/specs[source]["path"]).read_text())
        anchor, rows, proposal_audit = predict(folder, corpus, name, model)
        proposal_path = run/(name+".development-proposals.json")
        dump(proposal_path, dict(proposals=rows, audit=proposal_audit, trained_on=source,
            model_sha256=specs[source]["sha256"]))
        graph, audits[name] = apply_transactions(anchor, rows, model["margin"], cfg)
        audits[name]["proposal_audit"] = proposal_audit
        path = run/(name+".prediction.npz")
        with path.open("xb") as handle:
            np.savez_compressed(handle, coords=graph.coords, edges=graph.edges)
        dump(run/(name+".edits.json"), audits[name])
        predictions[name] = dict(sha256=digest(path), trained_on=source, model_sha256=specs[source]["sha256"],
            proposal_sha256=digest(proposal_path))
        print(json.dumps(dict(phase="ranker_development_prediction", dataset=name,
            selected=len(audits[name]["selected"]), status=audits[name]["status"])), flush=True)
    dump(run/"development-prediction-manifest.json", predictions)
    archive_rows = {r["dataset"]: r for r in archived["rows"]["reference"]}
    for name, info in predictions.items():
        path = run/(name+".prediction.npz")
        verify_file(path, info["sha256"])
        verify_file(run/(name+".development-proposals.json"), info["proposal_sha256"])
        with np.load(path, allow_pickle=False) as f:
            graph = Lineage(f["coords"], f["edges"]).validate()
        anchor, _, _, _ = previous.open_evidence(folder, name, corpus)
        np.testing.assert_array_equal(graph.coords, anchor.coords)
        row = previous.resolve_dataset_row(name, corpus["data_root"], split="development", gt=True)
        gt_hashes[name] = previous.tree_fingerprint(Path(row["gt"]))
        refs[name] = previous.metric_row(anchor, row)
        previous.assert_baseline(refs[name], archive_rows[name], 1e-10)
        predicted[name] = previous.metric_row(graph, row)
    evaluation = previous.evaluate_panel(refs, predicted)
    gate = previous.assess(evaluation, "development")
    if any(a["status"] == "rejected_edit_budget" for a in audits.values()):
        gate["passed"] = False
        gate["reasons"].append("Development edit budget rejected")
    for name, sha in gt_hashes.items():
        if previous.tree_fingerprint(Path(corpus["data_root"])/(name+".geff")) != sha:
            raise ValueError("Annotations changed during012")
    for spec in specs.values():
        verify_file(run/spec["path"], spec["sha256"])
    verify_sources(folder, code_manifest())
    result = dict(disposition=SCOPE, evaluation=evaluation, development_gate=gate, models=specs,
        edits=audits, gt_sha256=gt_hashes, source_unchanged=True, model_trained=True,
        threshold_selected=True, calibration_evaluated=True, margin_selection_scope="source_prefix_on_or_off_only",
        next_action="designer_review_only" if gate["passed"] else "rejected_no_kaggle_submission", **safety(True))
    dump(run/"ranking.json", result)
    return {k: v for k, v in result.items() if k not in {"evaluation", "edits", "gt_sha256"}} | {
        "ranking_sha256": digest(run/"ranking.json")}
