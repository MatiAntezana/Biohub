"""011 label-free local association selector. No annotation paths or GT inputs.

Scores are regularized discriminants, NOT calibrated probabilities. All terms
are evaluated on frozen reference features. The reference has gain zero; edits
pay for displaced edges, lost forks and new orphans. Non-overlapping transactions
are selected deterministically. This is a conservative greedy policy, not ILP.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from src.tracking.contracts import Lineage

EDGE_NAMES = ("distance", "axial_distance", "lateral_distance", "forward_residual",
    "mother_velocity_known", "backward_residual", "daughter_velocity_known",
    "patch_correlation", "patch_pair_valid", "log_contrast_difference",
    "min_contrast", "max_contrast", "min_inbounds", "reference_edge")
FORK_NAMES = ("min_distance", "max_distance", "daughter_separation", "cosine",
    "distance_asymmetry", "midpoint_distance", "axial_separation",
    "midpoint_forward_residual", "mother_velocity_known", "daughter_velocity_difference",
    "both_daughter_velocities_known", "min_mother_daughter_correlation",
    "max_mother_daughter_correlation", "daughter_correlation", "all_patches_valid",
    "log_contrast_sum_ratio", "daughter_contrast_asymmetry", "min_inbounds")


def topology(anchor):
    children, parents = defaultdict(set), {}
    for u, v in anchor.edges:
        children[int(u)].add(int(v))
        parents[int(v)] = int(u)
    return children, parents


def candidate_tables(anchor, scale, cfg):
    """Same radius/K as009; retain ALL original edges/forks as scored controls.

    New single-child options are restricted to daughters of existing forks:
    removing a spurious branch is in scope; general continuation relinking is not.
    """
    anchor.validate()
    c, s = anchor.coords, np.asarray(scale, float)
    if s.shape != (3,) or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Positive physical scale required")
    children, _ = topology(anchor)
    forks = {tuple([m, *sorted(ds)]) for m, ds in children.items() if len(ds) == 2}
    edges = set(map(tuple, anchor.edges.tolist()))
    for t in np.unique(c[:, 0]):
        mothers = np.flatnonzero(c[:, 0] == t)
        daughters = np.flatnonzero(c[:, 0] == t + 1)
        if len(daughters) < 2:
            continue
        tree = cKDTree(c[daughters, 1:] * s)
        near = tree.query_ball_point(c[mothers, 1:] * s, cfg["radius_um"])
        for m, hits in zip(mothers, near):
            ranked = sorted((float(np.linalg.norm((c[daughters[j], 1:] - c[m, 1:]) * s)),
                int(daughters[j])) for j in hits)[:cfg["max_children"]]
            ds = sorted(d for _, d in ranked)
            for a, b in combinations(ds, 2):
                forks.add((int(m), a, b))
                edges.update(((int(m), a), (int(m), b)))
            if len(forks) > cfg["max_pairs_per_dataset"]:
                raise ValueError("Selector proposal budget exceeded; no truncation")
    return (np.asarray(sorted(edges), np.int64).reshape(-1, 2),
            np.asarray(sorted(forks), np.int64).reshape(-1, 3))


def image_descriptors(anchor, source, cfg):
    """One measured, physically sampled patch at EACH node's own position/time.

    Zero padding is marked by in-bounds coverage. Flat patches have an explicit
    missing-contrast flag, never fabricated signal. No full movie is loaded.
    """
    anchor.validate(source.shape)
    shape, half = tuple(cfg["patch_shape"]), cfg["patch_half_um"]
    size = int(np.prod(shape))
    normalized = np.zeros((len(anchor.coords), size), np.float32)
    stats = np.zeros((len(anchor.coords), 3), np.float32)  # contrast, valid, coverage
    grid = np.stack(np.meshgrid(*[np.linspace(-half/s, half/s, n)
        for s, n in zip(source.scale, shape)], indexing="ij"), axis=-1).reshape(-1, 3)
    center = tuple(slice(n//2 - 1, n//2 + 2) for n in shape)
    for i in np.argsort(anchor.coords[:, 0], kind="stable"):
        t, z, y, x = anchor.coords[i]
        patch = source.patch(int(t), [z, y, x], half_um=half, shape=shape)
        if patch.shape != shape or not np.isfinite(patch).all():
            raise ValueError("Invalid measured image patch")
        inside = ((grid + [z, y, x] >= 0) & (grid + [z, y, x] <= np.asarray(source.shape[1:])-1)).all(axis=1)
        # Exclude padding from normalization and contrast statistics.
        values = patch.ravel().astype(float)
        if not inside.any():
            raise ValueError("Detection patch outside image")
        centred = np.where(inside, values - values[inside].mean(), 0.)
        norm = np.linalg.norm(centred)
        valid = norm > 1e-8
        if valid:
            normalized[i] = centred / norm
        central_mask = np.zeros(shape, bool)
        central_mask[center] = True
        central = central_mask.ravel() & inside
        contrast = max(0., float(values[central].mean() - np.median(values[inside])))
        stats[i] = [contrast, float(valid), float(inside.mean())]
    return normalized, stats


def context(anchor, scale, descriptors, stats):
    n = len(anchor.coords)
    s = np.asarray(scale, float)
    if (s.shape != (3,) or (s <= 0).any() or not np.isfinite(s).all()
            or descriptors.ndim != 2 or len(descriptors) != n or stats.shape != (n, 3)
            or not np.isfinite(descriptors).all() or not np.isfinite(stats).all()):
        raise ValueError("Invalid node evidence")
    children, parents = topology(anchor)
    previous, following = np.zeros((n, 3)), np.zeros((n, 3))
    prev_valid, next_valid = np.zeros(n), np.zeros(n)
    for v, u in parents.items():
        # Parent division is not a reliable pre-division velocity estimate.
        if len(children[u]) == 1:
            previous[v] = (anchor.coords[v, 1:] - anchor.coords[u, 1:]) * s
            prev_valid[v] = 1
    for u, ds in children.items():
        if len(ds) == 1:
            v = next(iter(ds))
            following[u] = (anchor.coords[v, 1:] - anchor.coords[u, 1:]) * s
            next_valid[u] = 1
    return dict(anchor=anchor, scale=s, descriptors=descriptors, stats=stats,
        previous=previous, following=following, prev_valid=prev_valid, next_valid=next_valid,
        children=children, parents=parents, original=set(map(tuple, anchor.edges)))


def edge_features(ids, ctx):
    ids = np.asarray(ids, np.int64).reshape(-1, 2)
    a, b = ids.T
    c, s = ctx["anchor"].coords, ctx["scale"]
    delta = (c[b, 1:] - c[a, 1:]) * s
    v, w = ctx["prev_valid"][a], ctx["next_valid"][b]
    p, q = ctx["descriptors"][a], ctx["descriptors"][b]
    sa, sb = ctx["stats"][a], ctx["stats"][b]
    return np.column_stack([np.linalg.norm(delta, axis=1)/12, np.abs(delta[:, 0])/12,
        np.linalg.norm(delta[:, 1:], axis=1)/12,
        np.linalg.norm(delta-ctx["previous"][a], axis=1)*v/12, v,
        np.linalg.norm(delta-ctx["following"][b], axis=1)*w/12, w,
        (p*q).sum(axis=1), sa[:, 1]*sb[:, 1],
        np.abs(np.log1p(sa[:, 0])-np.log1p(sb[:, 0])),
        np.minimum(sa[:, 0], sb[:, 0]), np.maximum(sa[:, 0], sb[:, 0]),
        np.minimum(sa[:, 2], sb[:, 2]),
        [float(tuple(e) in ctx["original"]) for e in ids]]).astype(np.float32)


def fork_features(ids, ctx):
    ids = np.asarray(ids, np.int64).reshape(-1, 3)
    m, a, b = ids.T
    c, s = ctx["anchor"].coords, ctx["scale"]
    va, vb = (c[a, 1:]-c[m, 1:])*s, (c[b, 1:]-c[m, 1:])*s
    da, db = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    mid = (va+vb)*.5
    p, q, r = ctx["descriptors"][m], ctx["descriptors"][a], ctx["descriptors"][b]
    sm, sa, sb = ctx["stats"][m], ctx["stats"][a], ctx["stats"][b]
    ca, cb = (p*q).sum(axis=1), (p*r).sum(axis=1)
    v, w = ctx["prev_valid"][m], ctx["next_valid"][a]*ctx["next_valid"][b]
    return np.column_stack([np.minimum(da, db)/12, np.maximum(da, db)/12,
        np.linalg.norm(va-vb, axis=1)/12, (va*vb).sum(axis=1)/np.maximum(da*db, 1e-6),
        np.abs(da-db)/12, np.linalg.norm(mid, axis=1)/12, np.abs(va[:, 0]-vb[:, 0])/12,
        np.linalg.norm(mid-ctx["previous"][m], axis=1)*v/12, v,
        np.linalg.norm(ctx["following"][a]-ctx["following"][b], axis=1)*w/12, w,
        np.minimum(ca, cb), np.maximum(ca, cb), (q*r).sum(axis=1),
        sm[:, 1]*sa[:, 1]*sb[:, 1], np.log1p(sa[:, 0]+sb[:, 0])-np.log1p(sm[:, 0]),
        np.abs(np.log1p(sa[:, 0])-np.log1p(sb[:, 0])),
        np.minimum(sm[:, 2], np.minimum(sa[:, 2], sb[:, 2]))]).astype(np.float32)


def linear_scores(model, x):
    stats = model["standardizer"]
    mean, scale, active, weights = (np.asarray(a) for a in
        (stats["mean"], stats["scale"], stats["active"], model["weights"]))
    if (x.ndim != 2 or any(a.shape != (x.shape[1],) for a in (mean, scale, active, weights))
            or not all(np.isfinite(a).all() for a in (x, mean, scale, weights)) or (scale <= 0).any()
            or not np.isfinite(model["intercept"]) or active.dtype != bool):
        raise ValueError("Invalid fitted selector")
    return (((x-mean)/scale)*active) @ weights + model["intercept"]


def score_tables(edge_ids, fork_ids, ctx, model, cfg):
    values = []
    for kind, ids, fn, names in (("edge", edge_ids, edge_features, EDGE_NAMES),
                                 ("fork", fork_ids, fork_features, FORK_NAMES)):
        if model[kind]["feature_names"] != list(names):
            raise ValueError("Selector feature schema drift")
        scores = np.empty(len(ids), float)
        for start in range(0, len(ids), cfg["feature_batch_size"]):
            stop = start + cfg["feature_batch_size"]
            scores[start:stop] = linear_scores(model[kind], fn(ids[start:stop], ctx))
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite selector scores")
        values.append({tuple(map(int, k)): float(v) for k, v in zip(ids, scores)})
    return values


def transaction(mother, daughters, ctx, edge_scores, fork_scores, cfg):
    """Gain of a local transaction in a frozen edge/fork/disappearance energy."""
    children, parents = ctx["children"], ctx["parents"]
    ids = np.asarray([mother, *daughters])
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or (ids < 0).any() or (ids >= len(ctx["anchor"].coords)).any()):
        raise ValueError("Integer local node IDs required")
    desired = set(map(int, daughters))
    if len(desired) not in (1, 2) or len(desired) != len(daughters):
        raise ValueError("One or two distinct daughters required")
    c = ctx["anchor"].coords
    if not (c[list(desired), 0] == c[mother, 0]+1).all():
        raise ValueError("Nonconsecutive repair")
    remove = {(mother, d) for d in children.get(mother, set())-desired}
    remove |= {(parents[d], d) for d in desired if d in parents and parents[d] != mother}
    add = {(mother, d) for d in desired-children.get(mother, set())}
    affected = {mother} | {p for p, _ in remove}
    new = {p: set(children.get(p, set())) for p in affected}
    for p, d in remove:
        new[p].remove(d)
    for p, d in add:
        new[p].add(d)
    gain = sum(edge_scores[e] for e in add)-sum(edge_scores[e] for e in remove)
    for p in affected:
        old_ds, new_ds = children.get(p, set()), new[p]
        if len(old_ds) == 2:
            gain -= fork_scores[(p, *sorted(old_ds))]
        if len(new_ds) == 2:
            gain += fork_scores[(p, *sorted(new_ds))]
        gain -= cfg["orphan_penalty"] * (float(not new_ds)-float(not old_ds))
    if not np.isfinite(gain):
        raise ValueError("Non-finite repair gain")
    return dict(mother=int(mother), daughters=sorted(desired), gain=float(gain),
        add=sorted(add), remove=sorted(remove), affected_parents=sorted(affected),
        touched_children=sorted({d for _, d in add | remove}))


def propose_transactions(ctx, edge_scores, fork_scores, cfg):
    """At most ONE best positive-gain non-reference option per mother.

    Tied alternatives abstain, rather than selecting a convenient child ID.
    This bounds exported proposals without selecting by annotations.
    """
    best, ties = {}, set()
    options = ((m, (a, b)) for m, a, b in sorted(fork_scores))
    def consider(m, ds):
        tx = transaction(m, ds, ctx, edge_scores, fork_scores, cfg)
        if not tx["add"] and not tx["remove"] or tx["gain"] <= 0:
            return
        old = best.get(m)
        if old is None or tx["gain"] > old["gain"] + 1e-10:
            best[m] = tx
            ties.discard(m)
        elif abs(tx["gain"]-old["gain"]) <= 1e-10:
            ties.add(m)
    for m, ds in options:
        consider(m, ds)
    for m, ds in sorted(ctx["children"].items()):
        if len(ds) == 2:
            for d in sorted(ds):
                consider(m, (d,))
    return [best[m] for m in sorted(best) if m not in ties]


def apply_transactions(anchor, proposals, margin, cfg):
    """Apply disjoint writes against intact reference; abstain with margin=None.

    The edit cap is a rejection guard, never a quota or top-N pruning policy.
    An overflowing policy returns the untouched reference and a rejection audit.
    """
    anchor.validate()
    if margin is not None and (not np.isfinite(margin) or margin < 0):
        raise ValueError("Finite nonnegative margin or explicit abstention required")
    used_p, used_d, selected = set(), set(), []
    if margin is not None:
        for tx in sorted(proposals, key=lambda r: (-r["gain"], r["mother"], r["daughters"])):
            if not np.isfinite(tx["gain"]):
                raise ValueError("Non-finite proposal gain")
            if tx["gain"] <= margin:
                continue
            ps, ds = set(tx["affected_parents"]), set(tx["touched_children"])
            if ps & used_p or ds & used_d:
                continue
            selected.append(tx)
            used_p |= ps
            used_d |= ds
    cap = max(1, int(len(anchor.coords)*cfg["max_edit_fraction"]))
    overflow = len(selected) > cap
    considered = len(selected)
    if overflow:
        selected = []
    original = set(map(tuple, anchor.edges))
    add = {tuple(e) for tx in selected for e in tx["add"]}
    remove = {tuple(e) for tx in selected for e in tx["remove"]}
    if not remove <= original or add & original or add & remove:
        raise ValueError("Invalid repair transaction")
    kept = [tuple(e) for e in anchor.edges if tuple(e) not in remove]
    graph = Lineage(anchor.coords.copy(), np.asarray(kept+sorted(add), np.int64).reshape(-1, 2)).validate()
    children, _ = topology(graph)
    if any(children.get(tx["mother"], set()) != set(tx["daughters"]) for tx in selected):
        raise ValueError("Selected hypothesis not realized by repair transaction")
    return graph, dict(status="rejected_edit_budget" if overflow else "applied" if selected else "abstained",
        margin=margin, proposed_mothers=len(proposals), considered_transactions=considered,
        edit_cap=cap, selected=selected, added_edges=len(add), removed_edges=len(remove),
        nodes_unchanged=True, uses_gt_for_edits=False)
