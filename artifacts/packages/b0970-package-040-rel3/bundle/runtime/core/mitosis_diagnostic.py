"""Read-only, development-only audit of the rejected frozen mitosis arm.

This is not an experiment, a source for training, or a publication gate. It
observes the existing scorer without changing candidate selection or thresholds.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from core.artifacts import ROOT, code_manifest, digest, dump
from core.nextgen.contracts import load_candidates, load_reference, verify_file

SOURCE_HASH = "6db6f0671fa7358bf961a5c2b7005438b03f05b16fd424ff086ee62e1fb2b607"
DRIVER_CHANGES = {"ops/biohub.py", "ops/verify.py"}


def verify_diagnostic_source(receipt, current_manifest):
    """Allow old artifacts only for this non-promotable audit; science is exact."""
    manifest = receipt["code_manifest"]
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if identity != SOURCE_HASH or receipt["code_hash"] != SOURCE_HASH:
        raise ValueError("Diagnostic source must be the verified 005 release")
    if receipt["task"] != "prepare" or receipt["status"] != "completed" or receipt.get("result", {}).get("ready_for_experiments") is not True:
        raise ValueError("Diagnostic requires the completed prepared source")
    changed = [p for p, sha in manifest.items() if p not in DRIVER_CHANGES and current_manifest.get(p) != sha]
    if set(manifest) - set(current_manifest):
        raise ValueError("Diagnostic release cannot delete source files")
    additions = set(current_manifest) - set(manifest) - {"core/mitosis_diagnostic.py", "tests/test_mitosis_diagnostic.py"}
    if changed or additions:
        raise ValueError(f"Diagnostic-only reuse forbids scientific code drift: {changed}, {sorted(additions)}")


def inventory(anchor, pairs, probabilities, scale, config):
    """Mirror frozen enumeration for coverage; runtime capture must agree exactly."""
    children, parents, outgoing = defaultdict(set), {}, defaultdict(list)
    for u, v in anchor.edges:
        children[int(u)].add(int(v))
        parents[int(v)] = int(u)
    for (u, v), p in zip(pairs, probabilities):
        if np.isfinite(p) and p >= config["min_edge_probability"]:
            if np.linalg.norm((anchor.coords[v, 1:] - anchor.coords[u, 1:]) * scale) <= config["radius_um"]:
                outgoing[int(u)].append((int(v), float(p)))
    triples, addable = [], set()
    for mother, rows in sorted(outgoing.items()):
        selected = [v for v, _ in sorted(rows, key=lambda r: (-r[1], r[0]))[:config["max_children"]]]
        for a, b in combinations(sorted(selected), 2):
            triple = (mother, a, b)
            triples.append(triple)
            if len(children[mother]) == 1 and children[mother].issubset({a, b}):
                other = next(iter({a, b} - children[mother]))
                if other not in parents:
                    addable.add(triple)
    return triples, addable


@contextmanager
def capture_scores(anchor):
    """Observe actual frozen batches and return the original logits unchanged."""
    from core.nextgen import mitosis
    original_load, original_example = mitosis.load_head, mitosis.example_tensors
    coord_ids = {tuple(c): i for i, c in enumerate(anchor.coords)}
    scores, pending = [], []

    def tensors(triple, source):
        result = original_example(triple, source)
        pending.append(tuple(coord_ids[tuple(c)] for c in triple))
        return result

    class ObservedHead:
        def __init__(self, real):
            self.real = real

        def __call__(self, *args, **kwargs):
            logits = self.real(*args, **kwargs)
            values = logits.detach().sigmoid().cpu().numpy()
            if len(values) != len(pending) or not np.isfinite(values).all():
                raise ValueError("Diagnostic capture batch mismatch/non-finite scores")
            scores.extend({"ids": ids, "score": float(value)} for ids, value in zip(pending, values))
            pending.clear()
            return logits

    try:
        mitosis.example_tensors = tensors
        mitosis.load_head = lambda *a, **kw: ObservedHead(original_load(*a, **kw))
        yield scores
        if pending:
            raise ValueError("Unscored diagnostic batch")
    finally:
        mitosis.example_tensors, mitosis.load_head = original_example, original_load


def score_summary(values, threshold):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite diagnostic scores")
    return {"count": len(values), "at_frozen_threshold": int((values >= threshold).sum()),
        "quantiles": dict(zip(("min", "p10", "p50", "p90", "p99", "max"),
            np.quantile(values, [0, .1, .5, .9, .99, 1]).tolist())) if len(values) else None}


def training_batch_summary(records, config):
    """Reconstruct batching, not training: no model or image is executed."""
    rng = np.random.default_rng(config["seed"])
    rows = []
    for epoch in range(config["epochs"]):
        shuffled = []
        sequences = sorted({r["dataset"] for r in records})
        rng.shuffle(sequences)
        for name in sequences:
            group = [r for r in records if r["dataset"] == name]
            rng.shuffle(group)
            shuffled.extend(group)
        counts = [sum(r["label"] for r in shuffled[i:i + config["batch_size"]])
                  for i in range(0, len(shuffled), config["batch_size"])]
        rows.append({"epoch": epoch + 1, "batches": len(counts),
                     "all_negative_batches": counts.count(0), "positive_counts": counts})
    return rows


def classify_divisions(gt_children, gt_to_anchor, anchor, scored, threshold):
    """Exact consecutive forks only, NOT the official windowed division metric."""
    children, parents = defaultdict(set), {}
    for u, v in anchor.edges:
        children[int(u)].add(int(v)); parents[int(v)] = int(u)
    output = []
    for mother, daughters in sorted(gt_children.items()):
        if len(daughters) != 2:
            continue
        ids = [gt_to_anchor.get(i) for i in (mother, *sorted(daughters))]
        score = None
        if any(i is None for i in ids):
            reason = "unmatched_endpoint"
        else:
            m, a, b = ids
            pair = {a, b}
            if children[m] == pair:
                reason = "already_present"
            elif len(children[m]) != 1:
                reason = "reference_mother_not_single"
            elif not children[m].issubset(pair):
                reason = "existing_child_not_true_pair"
            elif next(iter(pair - children[m])) in parents:
                reason = "second_daughter_already_assigned"
            else:
                score = scored.get((m, *sorted(pair)))
                reason = "not_scored" if score is None else "below_threshold" if score < threshold else "approved_before_competition"
        output.append({"gt_mother": int(mother), "gt_daughters": sorted(map(int, daughters)),
            "anchor_ids": ids, "reason": reason, "score": score})
    return output


def match_annotation(anchor, gt_path, scale):
    """Use the archived metric's full matching and preserve coordinate/ID mapping."""
    import sys
    import tracksdata as td
    from core.evaluation import make_graph, metric_root
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph
    sys.path.insert(0, str(metric_root()))
    from tracking_cellmot.division_metrics import _match_full
    nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(anchor.coords)}
    links = [{"source_id": int(u), "target_id": int(v)} for u, v in anchor.edges]
    pred = make_graph(*as_evaluator_inputs(nodes, links, round_coordinates=False))
    gt = load_graph(gt_path)
    matched = _match_full(pred, gt, tuple(scale), 7.0)
    coord_ids = {tuple(c): i for i, c in enumerate(anchor.coords)}
    gt_to_anchor = {}
    for row in matched.node_attrs().iter_rows(named=True):
        gid = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        if gid is not None and gid != -1:
            key = tuple(float(row[k]) for k in ("t", "z", "y", "x"))
            if int(gid) in gt_to_anchor:
                raise ValueError("Non-unique GT match in diagnostic")
            gt_to_anchor[int(gid)] = coord_ids[key]
    children, parents = defaultdict(set), {}
    for u, v in gt.edge_list():
        children[int(u)].add(int(v)); parents[int(v)] = int(u)
    attrs = {int(r["node_id"]): r for r in gt.node_attrs().iter_rows(named=True)}
    triples = {m: np.asarray([[attrs[i][k] for k in ("t", "z", "y", "x")]
        for i in (m, *sorted(ds))], float) for m, ds in children.items() if len(ds) == 2}
    return gt_to_anchor, children, parents, triples


def diagnostic_job(args, run, source):
    """No new detector/training/threshold search; confirmation cache never opened."""
    import torch
    from core.nextgen.candidates import build_candidates
    from core.nextgen.images import ImageSource
    from core.nextgen.mitosis import load_head, example_tensors, confirmed_label
    from core.nextgen.pipeline import infer_dataset
    from core.protocol import PROTOCOL, resolve_dataset_row
    source_receipt = json.loads((source / "receipt.json").read_text())
    verify_diagnostic_source(source_receipt, code_manifest())
    verify_file(source / "plan.json", source_receipt["result"]["plan_sha256"])
    parity = json.loads((source / "replay/parity.json").read_text())
    expected = json.loads((ROOT / "legacy/provenance.json").read_text())["expected_v19_csv_sha256"]
    if parity.get("passed") is not True or parity.get("actual_sha256") != expected or parity.get("expected_sha256") != expected:
        raise ValueError("Diagnostic source lacks exact historical parity")
    verify_file(source / "replay/replayed.csv", expected)
    config = json.loads((ROOT / "configs/nextgen.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    names = PROTOCOL["development"]
    if plan["panels"]["development"] != names:
        raise ValueError("Development panel drift")
    threshold = config["mitosis"]["threshold"]
    cache = source / "cache-development"
    cache_receipt = json.loads((cache / "cache.json").read_text())
    if cache_receipt["code_hash"] != SOURCE_HASH or sorted(cache_receipt["datasets"]) != sorted(names):
        raise ValueError("Incomplete or different development cache")
    verify_file(cache / "reference.csv", cache_receipt["reference_sha256"])
    reference = load_reference(cache / "reference.csv", names)
    models = json.loads((source / "models.json").read_text())
    if models != source_receipt["result"]["models"]:
        raise ValueError("Prepared model identities differ from completed receipt")
    reports, training = {}, {}
    for group in PROTOCOL["group_prefixes"]:
        folder = source / ("train-" + group)
        info = json.loads((folder / "training.json").read_text())
        verify_file(folder / "examples.jsonl", info["examples_sha256"])
        records = [json.loads(s) for s in (folder / "examples.jsonl").read_text().splitlines()]
        training[group] = {"history": info["history"], "class_balance": info["class_balance"],
            "batch_composition": training_batch_summary(records, config["training"])}
    dump(run / "training-audit.json", training)
    for name in sorted(names):
        anchor = reference[name]
        row = resolve_dataset_row(name, plan["data_root"], split="confirmation", gt=False)
        # 'confirmation' here enforces the existing OTHER-GROUP head guard;
        # this row belongs exclusively to the declared development panel.
        other = next(g for g in PROTOCOL["group_prefixes"] if g != row["group"])
        spec = {**models[other], "checkpoint": str(source / ("train-" + other) / "mitosis_head.pt")}
        row.update(candidates=str(cache / "raw" / (name + "_candidates.npz")),
                   candidates_receipt=str(cache / "raw" / (name + ".json")))
        raw_receipt = json.loads(Path(row["candidates_receipt"]).read_text())
        row["candidates_sha256"] = raw_receipt["candidates_sha256"]
        raw_coords, raw_edges = load_candidates(row)
        pairs, priors, candidate_report = build_candidates(anchor, raw_coords, raw_edges, row["scale_um"], config["candidates"])
        expected, addable = inventory(anchor, pairs, priors, row["scale_um"], config["mitosis"])
        with capture_scores(anchor) as scores:
            graph, details = infer_dataset(anchor, row, {"mitosis": spec}, config, "mitosis_add")
        if [r["ids"] for r in scores] != expected:
            raise ValueError("Audit enumeration differs from actual frozen scorer")
        # The rejected 005 output must remain a no-op. Never turn this audit into a candidate.
        if not np.array_equal(graph.coords, anchor.coords) or set(map(tuple, graph.edges)) != set(map(tuple, anchor.edges)):
            raise ValueError("Diagnostic replay no longer matches the rejected arm; return evidence")
        dump(run / (name + ".scores.json"), scores)
        # GT is first opened AFTER the original label-free inference has finished.
        mapping, gt_children, gt_parents, gt_triples = match_annotation(anchor,
            Path(plan["data_root"]) / (name + ".geff"), row["scale_um"])
        inverse = {v: k for k, v in mapping.items()}
        labelled = defaultdict(list)
        for rec in scores:
            ids = rec["ids"]
            gt_ids = [inverse.get(i) for i in ids]
            label = None if any(i is None for i in gt_ids) else confirmed_label(*gt_ids, gt_parents)
            key = {None: "unknown", 0: "confirmed_negative", 1: "confirmed_positive"}[label]
            labelled[key].append(rec["score"])
            if ids in addable:
                labelled["addable_" + key].append(rec["score"])
        classified = classify_divisions(gt_children, mapping, anchor,
            {r["ids"]: r["score"] for r in scores}, threshold)
        prior_lookup = {tuple(pair): float(p) for pair, p in zip(pairs, priors)}
        for fork in classified:
            if fork["reason"] == "not_scored":
                m, a, b = fork["anchor_ids"]
                ps = [prior_lookup.get((m, child), float("nan")) for child in (a, b)]
                distances = [float(np.linalg.norm((anchor.coords[child, 1:] - anchor.coords[m, 1:]) * row["scale_um"])) for child in (a, b)]
                fork["edge_priors"] = [p if np.isfinite(p) else None for p in ps]
                fork["distances_um"] = distances
                fork["candidate_block"] = ("missing_neural_prior" if not np.isfinite(ps).all()
                    else "edge_probability" if min(ps) < config["mitosis"]["min_edge_probability"]
                    else "radius" if max(distances) > config["mitosis"]["radius_um"] else "top_k")
        # Teacher-coordinate sanity check, development only, still the opposite head.
        head, image_source = load_head(spec, row, "cuda"), ImageSource(row["image"], "0", row["scale_um"])
        on_gt = []
        with torch.inference_mode():
            for mother, triple in sorted(gt_triples.items()):
                im, geom = example_tensors(triple, image_source)
                value = float(head(torch.as_tensor(im[None], device="cuda"),
                    torch.as_tensor(geom[None], device="cuda")).sigmoid().item())
                if not np.isfinite(value):
                    raise ValueError("Non-finite GT diagnostic score")
                on_gt.append({"gt_mother": int(mother), "score": value})
        del head, image_source
        reports[name] = {**candidate_report, "actual_scored_pairs": len(scores), "structurally_addable_pairs": len(addable),
            "scores_all": score_summary([r["score"] for r in scores], threshold),
            "scores_addable": score_summary([r["score"] for r in scores if r["ids"] in addable], threshold),
            "scores_by_annotation": {k: score_summary(v, threshold) for k, v in labelled.items()},
            "exact_gt_forks": classified, "exact_gt_fork_reasons": dict(Counter(r["reason"] for r in classified)),
            "development_gt_coordinate_scores": on_gt, "frozen_output": details, "head_sha256": spec["sha256"]}
        dump(run / (name + ".diagnostic.json"), reports[name])
        print(json.dumps({"dataset": name, "scored_pairs": len(scores), "addable_pairs": len(addable),
            "max_score": reports[name]["scores_all"]["quantiles"]}), flush=True)
        torch.cuda.empty_cache()
    result = {"disposition": "diagnostic_only_no_submission", "source_run": source.name,
        "source_code_hash": SOURCE_HASH, "datasets": reports, "frozen_threshold": threshold,
        "confirmation_evaluated": False, "scope": "development_exact_forks_not_official_windowed_division_score"}
    dump(run / "diagnostic.json", result)
    return {"disposition": result["disposition"], "datasets": len(reports),
        "diagnostic_sha256": digest(run / "diagnostic.json"), "confirmation_evaluated": False}
