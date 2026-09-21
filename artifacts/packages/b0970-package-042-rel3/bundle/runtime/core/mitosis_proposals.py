"""009: geometry on actual reference detections, without graph mutation.

Prediction has no GT inputs. Labels and exact-fork coverage are computed only
after every development prediction is frozen. Unknown is never a negative.
"""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

from core.artifacts import ROOT, code_manifest, digest, dump
from core.mitosis_probe import verify_probe_source, load_probe, predict_probe, probe_metrics, GEOMETRY_NAMES
from core.mitosis_diagnostic import inventory, match_annotation
from core.nextgen.candidates import build_candidates
from core.nextgen.contracts import load_candidates, load_reference, verify_file
from core.nextgen.mitosis import confirmed_label
from core.protocol import PROTOCOL, resolve_dataset_row

SCOPE = "proposal_assay_only_no_submission"
ADDITIONS = {"core/mitosis_proposals.py", "configs/mitosis_proposals.json", "tests/test_mitosis_proposals.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}


def config():
    return json.loads((ROOT / "configs/mitosis_proposals.json").read_text())


def verify_sources(source, manifest):
    cfg = config()
    previous = {p: s for p, s in manifest.items() if p not in ADDITIONS}
    science = {p: s for p, s in previous.items() if p not in DRIVERS}
    identity = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    if identity != cfg["parent_science_sha256"] or source.name != cfg["source_run"]:
        raise ValueError("Proposal assay requires exact 008 science and pinned 005 source")
    verify_probe_source(source, previous)
    for name, files in cfg["source_sha256"].items():
        for relative, sha in files.items():
            verify_file(source.parent / name / relative, sha)
    probe = source.parent / cfg["probe_run"]
    receipt = json.loads((probe / "receipt.json").read_text())
    if (receipt["status"] != "completed" or receipt["task"] != "probe-mitosis"
            or receipt["source_run"] != source.name
            or receipt["result"]["disposition"] != "representation_assay_only_no_submission"):
        raise ValueError("Expected the completed pinned representation assay")
    return probe


def geometry_features(triples, scale, frame_count):
    """Original eleven features, with no image reads and no GT-derived values."""
    triples, scale = np.asarray(triples, float), np.asarray(scale, float)
    if (triples.ndim != 3 or triples.shape[1:] != (3, 4) or scale.shape != (3,)
            or not np.isfinite(triples).all() or not np.isfinite(scale).all()
            or (scale <= 0).any() or (triples < 0).any()
            or frame_count < 1 or int(frame_count) != frame_count):
        raise ValueError("Invalid physical triples/frame count")
    times = triples[:, :, 0]
    if (not np.equal(times, np.floor(times)).all() or (times >= frame_count).any()
            or not (times[:, 1:] == times[:, :1] + 1).all()):
        raise ValueError("Triples must connect consecutive frames inside the image")
    va = (triples[:, 1, 1:] - triples[:, 0, 1:]) * scale
    vb = (triples[:, 2, 1:] - triples[:, 0, 1:]) * scale
    da, db = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    geometry = np.column_stack([np.minimum(da, db)/12, np.maximum(da, db)/12,
        np.linalg.norm(va-vb, axis=1)/12, np.sum(va*vb, axis=1)/np.maximum(da*db, 1e-6),
        np.abs(da-db)/12, np.linalg.norm((va+vb)*.5, axis=1)/12, np.abs(va[:, 0]-vb[:, 0])/12])
    t = times[:, :1] + np.arange(-1, 3)
    return np.concatenate([geometry, ((t >= 0) & (t < frame_count))], axis=1).astype(np.float32)


def enumerate_pairs(anchor, scale, cfg):
    if cfg != config():
        raise ValueError("Only the fixed physical proposal recipe is allowed")
    anchor.validate()
    c, scale = anchor.coords, np.asarray(scale, float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Explicit positive physical spacing required")
    triples = []
    for t in np.unique(c[:, 0]):
        mothers, daughters = np.flatnonzero(c[:, 0] == t), np.flatnonzero(c[:, 0] == t+1)
        if len(daughters) < 2:
            continue
        tree = cKDTree(c[daughters, 1:] * scale)
        hits = tree.query_ball_point(c[mothers, 1:] * scale, cfg["radius_um"])
        for mother, near in zip(mothers, hits):
            distances = [(float(np.linalg.norm((c[daughters[i], 1:]-c[mother, 1:])*scale)), int(daughters[i])) for i in near]
            selected = [i for _, i in sorted(distances)[:cfg["max_children"]]]
            # No parent-occupancy, existing-child or neural-prior filter.
            triples.extend((int(mother), a, b) for a, b in combinations(sorted(selected), 2))
            if len(triples) > cfg["max_pairs_per_dataset"]:
                raise ValueError("Proposal memory budget exceeded; never silently truncate")
    return np.asarray(sorted(triples), dtype=np.int64).reshape(-1, 3)


def score_pairs(anchor, triples, scale, frame_count, model, batch_size):
    logits = []
    if model.get("variant") != "geometry" or model.get("feature_names") != list(GEOMETRY_NAMES):
        raise ValueError("Only the pinned geometry probe has compatible inference features")
    for start in range(0, len(triples), batch_size):
        features = geometry_features(anchor.coords[triples[start:start+batch_size]], scale, frame_count)
        logits.extend(predict_probe(model, features).tolist())
    logits = np.asarray(logits, float)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite geometric proposal scores")
    return logits


def topology_maps(anchor):
    children, parents = defaultdict(set), {}
    for u, v in anchor.edges:
        children[int(u)].add(int(v)); parents[int(v)] = int(u)
    return children, parents


def edit_footprint(triple, children, parents):
    """Describe necessary edits; NEVER apply them or claim biological correctness."""
    mother, a, b = map(int, triple)
    desired = {a, b}
    remove = {(mother, v) for v in children.get(mother, set()) - desired}
    remove.update((parents[v], v) for v in desired if v in parents and parents[v] != mother)
    add = {(mother, v) for v in desired - children.get(mother, set())}
    kind = ("already_present" if not add and not remove else "requires_reassignment" if remove
        else "additive_second_child" if len(add) == 1 else "new_two_child_fork")
    return {"kind": kind, "remove_edges": sorted(remove), "add_edges": sorted(add)}


def label_pairs(triples, gt_to_anchor, gt_parents):
    inverse = {v: k for k, v in gt_to_anchor.items()}
    if len(inverse) != len(gt_to_anchor):
        raise ValueError("Non-injective annotation matching")
    labels = []
    for triple in triples:
        ids = [inverse.get(int(i)) for i in triple]
        label = None if any(i is None for i in ids) else confirmed_label(*ids, gt_parents)
        labels.append(-1 if label is None else label)
    return np.asarray(labels, np.int8)


def annotation_summary(labels, logits, threshold):
    labels, logits = np.asarray(labels), np.asarray(logits)
    if (labels.shape != logits.shape or not set(labels.tolist()) <= {-1, 0, 1}
            or not np.isfinite(logits).all()):
        raise ValueError("Invalid proposal annotation arrays")
    known = labels >= 0
    return {"total_pairs": len(labels), "unknown_pairs": int((~known).sum()),
        "unknown_at_frozen_threshold": int(((labels == -1) & (expit(logits) >= threshold)).sum()),
        "confirmed_only_metrics": probe_metrics(labels[known], logits[known], threshold)}


def fork_coverage(anchor, triples, logits, legacy_triples, mapping, gt_children, gt_triples, scale, cfg):
    lookup = {tuple(map(int, triple)): i for i, triple in enumerate(triples)}
    per_mother = defaultdict(list)
    for i, (mother, _, _) in enumerate(triples):
        per_mother[int(mother)].append(i)
    legacy = set(map(tuple, legacy_triples))
    children, parents = topology_maps(anchor)
    rows = []
    for mother, daughters in sorted(gt_children.items()):
        if len(daughters) != 2:
            continue
        gt_ids = [int(mother), *sorted(map(int, daughters))]
        ids = [mapping.get(i) for i in gt_ids]
        row = {"gt_ids": gt_ids, "anchor_ids": ids, "covered": False, "legacy_scored": False}
        coords = gt_triples[mother]
        if not np.equal(coords[1:, 0], coords[0, 0]+1).all():
            row["reason"] = "nonconsecutive_gt_fork"
        elif any(i is None for i in ids):
            row["reason"] = "unmatched_endpoint"
        else:
            triple = (ids[0], *sorted(ids[1:]))
            distances = np.linalg.norm((anchor.coords[list(triple[1:]), 1:]-anchor.coords[triple[0], 1:])*scale, axis=1)
            row.update(legacy_scored=triple in legacy, distances_um=distances.tolist(),
                edit_footprint=edit_footprint(triple, children, parents))
            index = lookup.get(triple)
            if index is None:
                row["reason"] = "outside_radius" if max(distances) > cfg["radius_um"] else "outside_nearest_children"
            else:
                values = logits[per_mother[triple[0]]]
                value = logits[index]
                row.update(covered=True, reason="scored", logit=float(value), score=float(expit(value)),
                    rank_best=1+int((values > value).sum()), rank_worst=int((values >= value).sum()),
                    competing_pairs=len(values), at_frozen_threshold=bool(expit(value) >= cfg["threshold"]))
        rows.append(row)
    return rows


def tree_fingerprint(path):
    files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Empty annotation store")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def predict_dataset(source, name, plan, cfg):
    from core.nextgen.images import ImageSource
    cache = source / "cache-development"
    anchor = load_reference(cache / "reference.csv", [name])[name]
    row = resolve_dataset_row(name, plan["data_root"], split="confirmation", gt=False)
    # Metadata only: ImageSource.__init__ validates axes, units and shape; no pixels.
    shape = ImageSource(row["image"], "0", row["scale_um"]).shape
    anchor.validate(shape)
    other = next(g for g in PROTOCOL["group_prefixes"] if g != row["group"])
    relative = f"train-{other}/geometry.json"
    probe = source.parent / cfg["probe_run"]
    model = load_probe({"path": str(probe / relative), "sha256": cfg["source_sha256"][probe.name][relative]},
        row["group"], other, "geometry")
    raw_json = cache / "raw" / (name + ".json")
    info = json.loads(raw_json.read_text())
    row.update(candidates=str(cache / "raw" / (name + "_candidates.npz")),
        candidates_receipt=str(raw_json), candidates_sha256=info["candidates_sha256"])
    raw_coords, raw_edges = load_candidates(row)
    old_cfg = json.loads((ROOT / "configs/nextgen.json").read_text())
    pairs, priors, report = build_candidates(anchor, raw_coords, raw_edges, row["scale_um"], old_cfg["candidates"])
    legacy, _ = inventory(anchor, pairs, priors, row["scale_um"], old_cfg["mitosis"])
    triples = enumerate_pairs(anchor, row["scale_um"], cfg)
    logits = score_pairs(anchor, triples, row["scale_um"], shape[0], model, cfg["batch_size"])
    children, parents = topology_maps(anchor)
    topology = Counter(edit_footprint(triple, children, parents)["kind"] for triple in triples)
    return anchor, triples, logits, np.asarray(legacy, np.int64).reshape(-1, 3), {
        "dataset": name, "evaluated_data_group": row["group"], "probe_trained_on_group": other,
        "probe_sha256": cfg["source_sha256"][probe.name][relative], "shape": list(shape),
        "scale_um": row["scale_um"], "legacy_candidate_report": report,
        "legacy_scored_pairs": len(legacy), "geometric_pairs": len(triples), "topology_all_pairs": dict(topology)}


def proposal_job(args, run, source):
    verify_sources(source, code_manifest())
    cfg = config()
    plan = json.loads((source / "plan.json").read_text())
    if plan["panels"]["development"] != PROTOCOL["development"]:
        raise ValueError("Development panel drift")
    names = sorted(PROTOCOL["development"])
    identities, reports = {}, {}
    # Freeze ALL label-free predictions before ANY annotations are opened.
    for name in names:
        anchor, triples, logits, legacy, info = predict_dataset(source, name, plan, cfg)
        path = run / (name + ".proposals.npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, triples=triples, logits=logits, legacy_triples=legacy)
        identities[name] = digest(path)
        reports[name] = info
        dump(run / (name + ".prediction.json"), {**info, "proposals_sha256": identities[name]})
        print(json.dumps({"phase": "prediction", "dataset": name, "pairs": len(triples)}), flush=True)
    dump(run / "prediction-manifest.json", identities)
    # Existing matching code is unchanged. Labels remain separate from predictions.
    by_group = defaultdict(list)
    for name in names:
        info = reports[name]
        path = run / (name + ".proposals.npz")
        verify_file(path, identities[name])
        with np.load(path, allow_pickle=False) as arrays:
            triples, logits, legacy = (arrays[k] for k in ("triples", "logits", "legacy_triples"))
        anchor = load_reference(source / "cache-development/reference.csv", [name])[name]
        gt_path = Path(plan["data_root"]) / (name + ".geff")
        gt_sha = tree_fingerprint(gt_path)
        mapping, children, parents, gt_triples = match_annotation(anchor, gt_path, info["scale_um"])
        if tree_fingerprint(gt_path) != gt_sha:
            raise ValueError("Annotations changed during matching")
        labels = label_pairs(triples, mapping, parents)
        forks = fork_coverage(anchor, triples, logits, legacy, mapping, children, gt_triples,
            np.asarray(info["scale_um"]), cfg)
        with (run / (name + ".labels.npz")).open("xb") as stream:
            np.savez_compressed(stream, labels=labels)
        reports[name].update(annotation=annotation_summary(labels, logits, cfg["threshold"]),
            gt_sha256=gt_sha, exact_gt_forks=forks,
            coverage_reasons=dict(Counter(r["reason"] for r in forks)))
        dump(run / (name + ".diagnostic.json"), reports[name])
        by_group[info["evaluated_data_group"]].append((labels, logits, forks))
        verify_file(path, identities[name])
        print(json.dumps({"phase": "annotation", "dataset": name,
            "covered_gt_forks": sum(r["covered"] for r in forks), "gt_forks": len(forks)}), flush=True)
    summaries = {}
    for group, rows in by_group.items():
        labels, logits = (np.concatenate([r[i] for r in rows]) for i in (0, 1))
        forks = [f for r in rows for f in r[2]]
        summaries[group] = {**annotation_summary(labels, logits, cfg["threshold"]),
            "exact_gt_forks": len(forks), "covered_gt_forks": sum(f["covered"] for f in forks),
            "legacy_covered_gt_forks": sum(f["legacy_scored"] for f in forks),
            "unique_top1_gt_forks": sum(f.get("rank_worst") == 1 for f in forks),
            "covered_nonpresent_gt_forks": sum(f["covered"] and f.get("edit_footprint", {}).get("kind") != "already_present" for f in forks),
            "coverage_reasons": dict(Counter(f["reason"] for f in forks))}
    verify_sources(source, code_manifest())
    result = {"disposition": SCOPE, "source_run": source.name, "probe_run": cfg["probe_run"],
        "recipe": cfg, "prediction_sha256": identities, "datasets": reports, "by_group": summaries,
        "source_unchanged": True, "confirmation_evaluated": False, "tracking_evaluated": False,
        "graph_modified": False, "threshold_selected": False, "kaggle_score": None,
        "scope": "development_reference_coordinates_exact_forks_not_official_tracking_score"}
    dump(run / "proposals.json", result)
    return {"disposition": SCOPE, "proposals_sha256": digest(run / "proposals.json"),
        "source_unchanged": True, "confirmation_evaluated": False, "tracking_evaluated": False,
        "graph_modified": False, "threshold_selected": False}
