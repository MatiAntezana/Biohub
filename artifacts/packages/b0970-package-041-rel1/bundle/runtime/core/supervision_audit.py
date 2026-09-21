"""014 offline supervision diagnostics. Never fits, scores or edits a lineage.

Annotations are evidence for this audit only. A nearest detection is NOT a new
matching, nor proof that two positions represent the same biological cell.
"""
from collections import Counter

import numpy as np

from core import event_ranking as events
from core.nextgen import event_ranker as hypotheses, repair_ranker as features


def signatures(x, signs, keys):
    """Exact signed feature preference, scoped to one frozen dataset:mother.

    Opposite preferences stay distinct. Signed zero is canonicalized, but no
    rounding/tolerance merges similar vectors or different biological events.
    """
    x, signs, keys = np.asarray(x, float), np.asarray(signs), np.asarray(keys)
    if (x.shape != (len(keys), len(features.NAMES)) or signs.shape != keys.shape
            or keys.ndim != 1 or not np.isfinite(x).all()
            or not set(signs.tolist()) <= {-1, 1}
            or any(not isinstance(k, str) or not k.rpartition(":")[2].isdigit() for k in keys)):
        raise ValueError("Invalid signed preference table")
    signed = x * signs[:, None]
    signed[signed == 0] = 0.
    return [(str(k), v.astype("<f8").tobytes()) for k, v in zip(keys, signed)]


def deduplicate(x, signs, keys, prior=()):
    prior, seen = set(prior), set()
    labels, keep = [], []
    for i, signature in enumerate(signatures(x, signs, keys)):
        label = ("duplicate_prior" if signature in prior else
                 "duplicate_within_batch" if signature in seen else
                 "zero_information" if not np.any(np.asarray(x)[i]) else "novel")
        labels.append(label)
        if label == "novel":
            keep.append(i)
        seen.add(signature)
    return np.asarray(keep, np.int64), dict(counts=dict(Counter(labels)), row_status=labels,
        comparisons=len(labels), novel_comparisons=len(keep),
        novel_mothers=len({str(keys[i]) for i in keep}),
        novel_volumes=len({str(keys[i]).rsplit(":", 1)[0] for i in keep}))


def coordinates(value):
    value = np.asarray(value, float)
    if (value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all()
            or (value < 0).any() or not (value[:, 0] == np.rint(value[:, 0])).all()):
        raise ValueError("Invalid tzyx detection coordinates")
    return value


def nearest(coords, point, scale, radius):
    ids = np.flatnonzero(coords[:, 0] == point[0])
    if not len(ids):
        return dict(nearest_id=None, distance_um=None, within_radius=0, nearest_ties=[])
    distance = np.linalg.norm((coords[ids, 1:] - point[1:]) * scale, axis=1)
    best = int(np.argmin(distance))
    return dict(nearest_id=int(ids[best]), distance_um=float(distance[best]),
        within_radius=int((distance <= radius).sum()),
        nearest_ties=ids[distance == distance[best]].tolist())


def endpoint(gt_id, point, mapping, anchor, raw, scale, radius):
    a, b = nearest(anchor, point, scale, radius), nearest(raw, point, scale, radius)
    matched = mapping.get(gt_id)
    inverse = {v: k for k, v in mapping.items()}
    reason = ("matched" if matched is not None else "near_anchor_unmatched" if a["within_radius"]
              else "near_raw_but_no_near_anchor" if b["within_radius"]
              else "no_near_detection_in_saved_candidates")
    return dict(gt_id=int(gt_id), gt_tzyx=point.tolist(), matched_anchor_id=matched,
        matched_distance_um=None if matched is None else float(np.linalg.norm(
            (anchor[matched, 1:]-point[1:])*scale)), anchor=a, raw=b, category=reason,
        nearest_anchor_assigned_to_gt=inverse.get(a["nearest_id"]),
        interpretation="proximity_evidence_not_identity_or_causal_stage_attribution")


def net_edges(good, other):
    ga, gr, oa, ore = [set(map(tuple, t[k])) for t, k in
        ((good, "add"), (good, "remove"), (other, "add"), (other, "remove"))]
    return (ga-oa) | (ore-gr), (gr-ore) | (oa-ga)


def unknown_reasons(good, other, mapping, parents):
    inverse = {int(v): int(k) for k, v in mapping.items()}
    added, removed = net_edges(good, other)
    result = Counter()
    for a, b in added | removed:
        if a not in inverse:
            result["unmapped_parent_detection"] += 1
        if b not in inverse:
            result["unmapped_child_detection"] += 1
        elif inverse[b] not in parents:
            result["child_gt_parent_unannotated"] += 1
    return result


def coverage(name, anchor, raw, forks, ctx, mapping, children, parents, triples, cfg):
    """Partition exact GT forks, using the unchanged011 candidate universe.

    No orphan is relabeled negative, no GT-based alternative is inserted. Rank
    diagnostics describe the frozen radius/K rule, including ID tie breaking.
    """
    c, raw = coordinates(anchor.coords), coordinates(raw)
    scale = np.asarray(ctx["scale"], float)
    if (scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any()
            or cfg["matching_radius_um"] <= 0):
        raise ValueError("Invalid audit geometry")
    if len(set(mapping.values())) != len(mapping) or any(v < 0 or v >= len(c) for v in mapping.values()):
        raise ValueError("Invalid/non-injective frozen matching")
    ids = hypotheses.hypothesis_ids(forks, ctx)
    truth, rows = events.edge_truth(mapping, parents), []
    for mother, ds in sorted(children.items()):
        if len(ds) != 2:
            continue
        gt_ids = [mother, *sorted(ds)]
        points = coordinates(triples[mother])
        if points.shape != (3, 4):
            raise ValueError("Missing exact GT triple")
        if any(g in mapping and c[mapping[g], 0] != point[0] for g, point in zip(gt_ids, points)):
            raise ValueError("GT/anchor frame mismatch")
        ep = [endpoint(g, q, mapping, c, raw, scale, cfg["matching_radius_um"])
              for g, q in zip(gt_ids, points)]
        row = dict(dataset=name, gt_mother=int(mother), biological_event=name+":"+str(mother),
                   endpoints=ep, stage="unmatched_endpoint", comparisons={}, unknown_reasons={})
        if all(g in mapping for g in gt_ids):
            target = (mapping[mother], *sorted(mapping[d] for d in ds))
            local = ids[ids[:, 0] == target[0]]
            pool = set(map(tuple, local))
            row.update(target=list(target), event=name+":"+str(target[0]),
                       stage="outside_frozen_pool", candidate_limits=[])
            daughter_ids = np.flatnonzero(c[:, 0] == c[target[0], 0]+1)
            ranked = sorted((float(np.linalg.norm((c[d, 1:]-c[target[0], 1:])*scale)),
                             int(d)) for d in daughter_ids)
            ranks = {d: i+1 for i, (_, d) in enumerate(ranked)}
            for d in target[1:]:
                distance = float(np.linalg.norm((c[d, 1:]-c[target[0], 1:])*scale))
                row["candidate_limits"].append(dict(daughter=d, distance_um=distance,
                    rank_all_next_frame=ranks.get(d), outside_radius=distance > cfg["radius_um"],
                    outside_top_k=ranks.get(d, float("inf")) > cfg["max_children"]))
            if target in pool:
                good = hypotheses.transaction(target, ctx)
                alternatives = [hypotheses.transaction(i, ctx) for i in local if tuple(i) != target]
                if (good["add"] or good["remove"]) and not any(
                        not t["add"] and not t["remove"] for t in alternatives):
                    alternatives.append(hypotheses.reference(target[0]))
                counts, missing = Counter(), Counter()
                for other in alternatives:
                    label = events.contrast_label(good, other, truth)
                    counts[{1: "accepted", -1: "contradictory", 0: "mixed", 2: "unknown", 3: "same_graph"}[label]] += 1
                    if label == events.UNKNOWN:
                        missing.update(unknown_reasons(good, other, mapping, parents))
                row.update(stage="usable_event" if counts["accepted"] else "covered_without_confirmed_contrast",
                    comparisons=dict(counts), unknown_reasons=dict(missing),
                    reference_correct=not bool(good["add"] or good["remove"]))
            elif not any(v["outside_radius"] or v["outside_top_k"] for v in row["candidate_limits"]):
                raise ValueError("Missing fork not explained by frozen candidate limits")
        rows.append(row)
    return dict(dataset=name, gt_forks=len(rows), stages=dict(Counter(r["stage"] for r in rows)),
        endpoint_categories=dict(Counter(e["category"] for r in rows for e in r["endpoints"])),
        events=rows, matching_radius_um=cfg["matching_radius_um"],
        raw_scope="saved_post_detector_candidates_before_global_ILP_and_postprocessing",
        exact_removal_stage_identified=False, matching_modified=False, graph_modified=False)
