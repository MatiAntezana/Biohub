"""014: three explicitly exposed training samples and a no-fit coverage audit.

Frozen005-013 science is read-only. Preparation uses the unmodified champion on
ONLY the allowlisted images; CPU analysis opens ONLY training annotations. No
margin selection, graph repair, validation score or submission is implemented.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core import event_ranking as old, supervision_audit as audit
from core.nextgen.contracts import load_reference, verify_file

ADDITIONS = {"core/coverage_audit.py", "core/supervision_audit.py", "configs/coverage_audit.json",
             "tests/test_coverage_audit.py", "ops/audit_coverage_inputs.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
PREP_SCOPE = "exposed_training_evidence_only_no_submission"
SCOPE = "coverage_supervision_audit_only_no_submission"
FALSE_FLAGS = ("model_trained", "calibration_evaluated", "development_evaluated",
    "confirmation_evaluated", "visible_evaluated", "tracking_evaluated", "threshold_selected",
    "threshold_selection_attempted", "uses_gt_for_edits", "kaggle_submission_allowed", "promoted")


def config():
    return json.loads((ROOT/"configs/coverage_audit.json").read_text())


def parent_manifest(current, archived):
    if hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest() != config()["source_release"]:
        raise ValueError("Unverified archived013 manifest")
    if ({k: v for k, v in current.items() if k not in ADDITIONS | DRIVERS}
            != {k: v for k, v in archived.items() if k not in DRIVERS}):
        raise ValueError("Unreviewed change to science/configuration001-013")
    return archived


def expansion_names(plan, corpus, cfg):
    names = sorted(cfg["exposed_training_only"])
    reserved = {n for panel in plan["panels"].values() for n in panel}
    historical = {n for group in corpus["splits"].values() for panel in group.values() for n in panel}
    if (not names or len(plan["all_datasets"]) != len(set(plan["all_datasets"]))
            or not set(names) <= set(old.source.PROTOCOL["previously_exposed"])
            or not set(names) <= set(plan["all_datasets"])
            or set(names) & (reserved | historical | set(corpus["development"]))
            or any(not n.startswith("44b6_") for n in names)
            or any(plan["public_info"][n]["divisions"] != cfg["exposed_training_only"][n] for n in names)):
        raise ValueError("Unapproved expansion, overlap or public annotation count drift")
    return names


def load_preferences(folder, group):
    with np.load(folder/("preferences-"+group+".npz"), allow_pickle=False) as f:
        return f["differences"], f["signs"], f["mother_keys"]


def historical_novelty(runs):
    result = {}
    for group in old.source.PROTOCOL["group_prefixes"]:
        x, y, keys = load_preferences(runs/old.config()["source_run"], group)
        with np.load(runs/config()["source_run"]/("training-"+group+".npz"), allow_pickle=False) as f:
            if set(f.files) != {"x", "y", "keys", "strata"}:
                raise ValueError("013 training schema drift")
            direct, event = f["strata"] != "event", f["strata"] == "event"
            if (not np.array_equal(f["x"][direct], x) or not np.array_equal(f["y"][direct], y)
                    or not np.array_equal(f["keys"][direct], keys)):
                raise ValueError("013 direct rows differ from012")
            _, result[group] = audit.deduplicate(f["x"][event], f["y"][event], f["keys"][event],
                                                audit.signatures(x, y, keys))
    return result


def audit_inputs(runs, plan_path, manifest=None):
    """Local safe audit: metadata, hashes and saved vectors. No images/GT/fits."""
    cfg, runs, plan_path = config(), Path(runs), Path(plan_path)
    folder = runs/cfg["source_run"]
    for name, sha in cfg["source_artifacts"].items():
        verify_file(folder/name, sha)
    receipt = json.loads((folder/"receipt.json").read_text())
    archived = parent_manifest(code_manifest() if manifest is None else manifest, receipt["code_manifest"])
    report = json.loads((folder/"event-ranking.json").read_text())
    if (receipt.get("status") != "completed" or receipt.get("task") != "learn-events"
            or receipt.get("node") != cfg["node"] or receipt.get("code_hash") != cfg["source_release"]
            or receipt["result"].get("event_ranking_sha256") != digest(folder/"event-ranking.json")
            or report.get("next_action") != "insufficient_supervision_no_submission"
            or report.get("source_unchanged") is not True
            or any(report.get(k) is not False for k in FALSE_FLAGS)
            or report.get("graph_exported") is not False):
        raise ValueError("Expected unchanged no-fit013 outcome")
    _, corpus, _ = old.audit_inputs(runs, archived)
    verify_file(plan_path, cfg["plan_sha256"])
    plan = json.loads(plan_path.read_text())
    names = expansion_names(plan, corpus, cfg)
    fit = {n for g in corpus["splits"].values() for n in g["fit"]}
    if set(cfg["raw_candidate_sha256"]) != fit:
        raise ValueError("Only all35 historical FIT raw candidates may enter014")
    for n, sha in cfg["raw_candidate_sha256"].items():
        verify_file(runs/cfg["corpus_run"] / "cache-training/raw" / (n+"_candidates.npz"), sha)
    result = dict(passed=True, source_release=cfg["source_release"], current_code_hash=code_hash(),
        evidence_npz=53, historical_fit_samples=len(fit), historical_raw_npz=len(fit),
        expansion=names, expansion_annotated_divisions=sum(cfg["exposed_training_only"].values()),
        previous_exposure_declared=True, historical_novelty=historical_novelty(runs),
        gt_opened=False, images_opened=False, historical005_chain_verified=False)
    return result, corpus, archived, plan


def verify_sources(folder, manifest):
    cfg = config()
    if folder.name != cfg["source_run"]:
        raise ValueError("014 preparation requires pinned b0970-events-013")
    result, corpus, archived, plan = audit_inputs(folder.parent,
        folder.parent/cfg["baseline_run"]/"plan.json", manifest)
    checked = old.verify_sources(folder.parent/old.config()["source_run"], archived)
    if checked != corpus:
        raise ValueError("Historical corpus changed during verification")
    result["historical005_chain_verified"] = True
    return result, corpus, plan


def payload_files(folder, names):
    """Hash only preparation outputs actually consumed by014; no live worker log."""
    relative = ["cache-training/reference.csv", "cache-training/cache.json"]
    for name in names:
        relative.extend([name+".evidence.npz", "cache-training/raw/"+name+"_candidates.npz",
                         "cache-training/raw/"+name+".json"])
    return {name: digest(folder/name) for name in relative}


def prepare_job(args, run, folder):
    from core.nextgen.images import ImageSource
    from ops.workflow import cache_panel
    before, historical, plan = verify_sources(folder, code_manifest())
    cfg = config()
    names = expansion_names(plan, historical, cfg)
    if shutil.disk_usage(run).free < cfg["minimum_free_disk_gib"]*1024**3:
        raise RuntimeError("014 preparation needs24 GiB free; preserve all previous runs")
    dump(run/"input-audit.json", before)
    dump(run/"split.json", dict(training_only=names, prior_exposure=True,
        reserved_panels_excluded=True, recipe=cfg, frozen011_recipe=old.source.config()))
    # Ground truth paths are never passed to Production/ImageSource.
    cache_panel(names, plan["data_root"], Path(args.artifact_root)/"assets/champion", run/"cache-training")
    anchors = load_reference(run/"cache-training/reference.csv", names)
    datasets = {}
    for name in names:
        row = old.source.resolve_dataset_row(name, plan["data_root"], split="test", gt=False)
        image = ImageSource(row["image"], "0", row["scale_um"])
        datasets[name] = old.source.save_evidence(run/(name+".evidence.npz"), anchors[name], image, old.source.config())
        datasets[name].update(group="44b6", panel="previously_exposed_training_only", image=row["image"])
        del image
        dump(run/"progress.json", dict(completed=sorted(datasets), total=len(names)))
        print(json.dumps(dict(phase="coverage_corpus", dataset=name)), flush=True)
    after, _, _ = verify_sources(folder, code_manifest())
    if before != after:
        raise ValueError("014 source changed during image preparation")
    corpus = dict(schema="exposed_training_corpus_014", disposition=PREP_SCOPE,
        code_hash=code_hash(), source_run=folder.name, recipe=cfg, data_root=plan["data_root"],
        splits={"44b6": dict(fit=names, calibration=[])}, datasets=datasets,
        previous_exposure_declared=True, training_only=True, gt_opened=False,
        source_unchanged=True, detector_full_fit=True, baseline_graph_exported=True,
        graph_edits_applied=False, payload_sha256=payload_files(run, names),
        **dict.fromkeys(FALSE_FLAGS, False))
    dump(run/"corpus.json", corpus)
    return dict(disposition=PREP_SCOPE, corpus_sha256=digest(run/"corpus.json"),
        ready_for_coverage_audit=True, datasets=len(names), source_unchanged=True,
        previous_exposure_declared=True, gt_opened=False, **dict.fromkeys(FALSE_FLAGS, False))


def verify_corpus(folder, manifest):
    cfg = config()
    if folder.name != cfg["prepare_run"]:
        raise ValueError("014 audit requires the exact new three-sample preparation")
    receipt = json.loads((folder/"receipt.json").read_text())
    expected = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (receipt.get("task") != "prepare-coverage" or receipt.get("status") != "completed"
            or receipt.get("code_hash") != expected or receipt.get("code_manifest") != manifest
            or receipt.get("node") != cfg["node"]
            or receipt.get("source_run") != cfg["source_run"]
            or receipt.get("result", {}).get("ready_for_coverage_audit") is not True):
        raise ValueError("Completed current-release014 preparation required on A1555")
    verify_file(folder/"corpus.json", receipt["result"]["corpus_sha256"])
    corpus = json.loads((folder/"corpus.json").read_text())
    before, historical, plan = verify_sources(folder.parent/cfg["source_run"], manifest)
    names = expansion_names(plan, historical, cfg)
    if (corpus.get("schema") != "exposed_training_corpus_014" or corpus.get("disposition") != PREP_SCOPE
            or corpus.get("code_hash") != expected or corpus.get("source_run") != cfg["source_run"]
            or corpus.get("recipe") != cfg or corpus.get("data_root") != plan["data_root"]
            or corpus.get("splits") != {"44b6": dict(fit=names, calibration=[])}
            or set(corpus["datasets"]) != set(names)
            or any(corpus.get(k) is not False for k in (*FALSE_FLAGS, "gt_opened", "graph_edits_applied"))
            or any(corpus.get(k) is not True for k in ("training_only", "previous_exposure_declared", "source_unchanged"))
            or corpus["payload_sha256"] != payload_files(folder, names)):
        raise ValueError("014 corpus partition, safety or payload drift")
    for name, info in corpus["datasets"].items():
        if info["group"] != "44b6" or info["panel"] != "previously_exposed_training_only":
            raise ValueError("014 exposure provenance drift")
        verify_file(folder/(name+".evidence.npz"), info["sha256"])
    return before, historical, corpus, plan


def verify_dataset_result(result, stats):
    stages = result["stages"]
    if (stages.get("unmatched_endpoint", 0) != stats.get("unmatched_gt_fork", 0)
            or stages.get("outside_frozen_pool", 0) != stats.get("outside_frozen_pool", 0)
            or stages.get("usable_event", 0) != stats.get("usable_events", 0)
            or sum(r["comparisons"].get("accepted", 0) for r in result["events"]) != stats.get("accepted", 0)):
        raise ValueError("Coverage accounting disagrees with frozen013 event definitions")


def dataset_audit(folder, corpus, name, expected_gt, run, cfg):
    anchor, _, forks, ctx = old.source.open_evidence(folder, name, corpus)
    with np.load(folder/"cache-training/raw"/(name+"_candidates.npz"), allow_pickle=False) as f:
        raw = f["coords"]
    gt = Path(corpus["data_root"])/(name+".geff")
    before = old.source.tree_fingerprint(gt)
    if before != expected_gt:
        raise ValueError("Training GT changed since archived preference extraction")
    mapping, children, parents, triples = old.source.match_annotation(anchor, gt, ctx["scale"])
    result = audit.coverage(name, anchor, raw, forks, ctx, mapping, children, parents, triples,
        {**old.source.config(), "matching_radius_um": cfg["matching_radius_um"]})
    x, rows, stats = old.event_examples(name, forks, ctx, mapping, children, parents)
    verify_dataset_result(result, stats)
    if old.source.tree_fingerprint(gt) != before:
        raise ValueError("Training GT changed during coverage audit")
    result.update(gt_sha256=before, event_stats=stats, confirmed_comparisons=rows)
    dump(run/(name+".coverage.json"), result)
    print(json.dumps(dict(phase="coverage_audit", dataset=name, stages=result["stages"])), flush=True)
    return result, x, rows


def audit_job(args, run, folder):
    cfg = config()
    before, historical, expanded, plan = verify_corpus(folder, code_manifest())
    dump(run/"input-audit.json", before)
    # Newly extracted DIRECT comparisons use the exact012 training-only label rule.
    # No optimizer is called; keep their bytes separate from the immutable012 run.
    direct_dir = run/"new-direct"
    direct_dir.mkdir()
    nx, ny, nk, gt_meta, direct_stats = old.old.collect_group(folder, expanded, "44b6", direct_dir)
    gx, gy, gk = load_preferences(folder.parent/old.config()["source_run"], "44b6")
    historical_signatures = audit.signatures(gx, gy, gk)
    keep_direct, direct_novelty = audit.deduplicate(nx, ny, nk, historical_signatures)
    prior = historical_signatures + audit.signatures(nx, ny, nk)
    reports, new_x, new_rows = {}, [], []
    for group in old.source.PROTOCOL["group_prefixes"]:
        meta = json.loads((folder.parent/cfg["source_run"]/("training-"+group+".json")).read_text())
        with np.load(folder.parent/cfg["source_run"]/("training-"+group+".npz"), allow_pickle=False) as f:
            historical_x, historical_keys, historical_strata = f["x"], f["keys"], f["strata"]
        for name in historical["splits"][group]["fit"]:
            result, x, rows = dataset_audit(folder.parent/cfg["corpus_run"], historical, name,
                meta["gt_sha256"][name], run, cfg)
            mask = (historical_strata == "event") & np.array([k.rsplit(":", 1)[0] == name for k in historical_keys], bool)
            if result["event_stats"] != meta["datasets"][name] or not np.array_equal(x, historical_x[mask]):
                raise ValueError("Historical event extraction drift; no silent recomputation")
            reports[name] = result
    for name in expanded["splits"]["44b6"]["fit"]:
        result, x, rows = dataset_audit(folder, expanded, name, gt_meta[name]["gt_sha256"], run, cfg)
        if result["gt_forks"] != cfg["exposed_training_only"][name]:
            raise ValueError("New sample annotation count differs from frozen metadata")
        reports[name] = result
        new_x.append(x)
        new_rows.extend(rows)
    ex = np.concatenate(new_x)
    ek = np.asarray([r["event"] for r in new_rows], str)
    keep_event, event_novelty = audit.deduplicate(ex, np.ones(len(ex)), ek, prior)
    _, event_vs012 = audit.deduplicate(ex, np.ones(len(ex)), ek, historical_signatures)
    signed = np.concatenate([nx[keep_direct]*ny[keep_direct, None], ex[keep_event]])
    keys = np.concatenate([np.asarray(nk, str)[keep_direct], ek[keep_event]])
    kinds = np.asarray(["direct"]*len(keep_direct)+["event"]*len(keep_event), str)
    tensor = run/"new-supervision-44b6.npz"
    with tensor.open("xb") as handle:
        np.savez_compressed(handle, signed_x=signed, keys=keys, kinds=kinds)
    # An audit has no promotion gate. Even useful new data requires design review.
    groups = {}
    for group in old.source.PROTOCOL["group_prefixes"]:
        groups[group] = {}
        for panel, names in (("historical_fit", historical["splits"][group]["fit"]),
                            ("previously_exposed_training", expanded["splits"].get(group, {}).get("fit", []))):
            counts = Counter()
            for n in names:
                counts.update(reports[n]["stages"])
            groups[group][panel] = dict(samples=len(names), gt_forks=sum(reports[n]["gt_forks"] for n in names),
                                       stages=dict(counts))
    after, _, checked, _ = verify_corpus(folder, code_manifest())
    if before != after or expanded != checked:
        raise ValueError("014 sources changed during audit")
    for name, report in reports.items():
        if old.source.tree_fingerprint(Path(plan["data_root"])/(name+".geff")) != report["gt_sha256"]:
            raise ValueError("Training annotation changed before audit finalization")
    result = dict(disposition=SCOPE, next_action="designer_review_no_submission", source_unchanged=True,
        dataset_count=len(reports), by_group=groups, previous_exposure_declared=True,
        detector_full_fit=True, independent_holdout=False, historical_novelty=before["historical_novelty"],
        new_direct_summary=direct_stats, new_direct_novelty=direct_novelty,
        new_direct_metadata_sha256=digest(direct_dir/"preferences-44b6.json"),
        new_direct_tensor_sha256=digest(direct_dir/"preferences-44b6.npz"),
        new_events_vs012=event_vs012, new_events_vs012_and_new_direct=event_novelty,
        new_event_comparisons=new_rows, new_unique_preferences=len(keys), new_unique_mothers=len(set(keys)),
        new_usable_biological_forks=len({(r["event"], r["gt_mother"]) for r in new_rows}),
        tensor_sha256=digest(tensor), tensor_schema="signed_preference_no_fit_training_only_v1",
        gt_sha256={n: r["gt_sha256"] for n, r in reports.items()},
        reports_sha256={n: digest(run/(n+".coverage.json")) for n in reports},
        graph_exported=False, graph_edits_applied=False, kaggle_score=None, **dict.fromkeys(FALSE_FLAGS, False))
    dump(run/"coverage.json", result)
    return {**result, "coverage_sha256": digest(run/"coverage.json")}
