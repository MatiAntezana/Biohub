"""012 annotation-free graph-potential differences and local repair ranking.

No fitted011 logits or fixed orphan penalty. A single learned potential values
ALL factors created and destroyed by a transaction. The intact reference is
exactly zero. Frozen measured descriptors and topology come from corpus011.
"""
from collections import defaultdict
from itertools import islice

import numpy as np

from core.nextgen import association as a

NAMES = tuple("edge_delta/"+n for n in a.EDGE_NAMES[:-1]) + tuple(
    "fork_delta/"+n for n in a.FORK_NAMES) + ("edge_count_delta", "fork_count_delta", "orphan_count_delta")


def batches(rows, size):
    if not isinstance(size, int) or size < 1:
        raise ValueError("Positive integer batch size required")
    rows = iter(rows)
    while batch := list(islice(rows, size)):
        yield batch


def transactions(forks, ctx):
    """All non-reference fork alternatives, plus removal of existing fork arms.

    No score/prior/occupancy filtering. The caller may restrict TRAINING rows,
    but inference always supplies the complete frozen proposal table.
    """
    zeros = defaultdict(float)
    for m, x, y in forks:
        tx = a.transaction(int(m), [int(x), int(y)], ctx, zeros, zeros, {"orphan_penalty": 0.})
        if tx["add"] or tx["remove"]:
            yield tx
    for m, ds in sorted(ctx["children"].items()):
        if len(ds) == 2:
            for d in sorted(ds):
                yield a.transaction(m, [d], ctx, zeros, zeros, {"orphan_penalty": 0.})


def changed_forks(tx, ctx):
    old = {p: set(ctx["children"].get(p, set())) for p in tx["affected_parents"]}
    new = {p: ds.copy() for p, ds in old.items()}
    for p, d in tx["remove"]:
        new[p].remove(d)
    for p, d in tx["add"]:
        new[p].add(d)
    old_forks = [(p, *sorted(ds)) for p, ds in old.items() if len(ds) == 2]
    new_forks = [(p, *sorted(ds)) for p, ds in new.items() if len(ds) == 2]
    orphan_delta = sum(not ds for ds in new.values()) - sum(not ds for ds in old.values())
    return old_forks, new_forks, orphan_delta


def features(rows, ctx):
    """Phi(repaired)-Phi(reference), including every displaced parent's factors.

    reference_edge is intentionally omitted: membership in the incumbent is not
    image evidence. Count/orphan coefficients are learned jointly, not hand-set.
    Work is bounded by a transaction batch, never a full-movie feature tensor.
    """
    changes = [changed_forks(tx, ctx) for tx in rows]
    edges = sorted({tuple(e) for tx in rows for e in tx["add"]+tx["remove"]})
    forks = sorted({f for old, new, _ in changes for f in old+new})
    ef = a.edge_features(edges, ctx)[:, :-1]
    ff = a.fork_features(forks, ctx)
    ei, fi = {e: i for i, e in enumerate(edges)}, {f: i for i, f in enumerate(forks)}
    x = np.zeros((len(rows), len(NAMES)), float)
    ne, nf = len(a.EDGE_NAMES)-1, len(a.FORK_NAMES)
    for i, (tx, (old, new, orphan)) in enumerate(zip(rows, changes)):
        for sign, es in ((1, tx["add"]), (-1, tx["remove"])):
            for e in es:
                x[i, :ne] += sign*ef[ei[tuple(e)]]
        for sign, fs in ((1, new), (-1, old)):
            for f in fs:
                x[i, ne:ne+nf] += sign*ff[fi[f]]
        x[i, -3:] = [len(tx["add"])-len(tx["remove"]), len(new)-len(old), orphan]
    if not np.isfinite(x).all():
        raise ValueError("Non-finite repair differences")
    return x


def scores(model, x):
    if (model.get("feature_names") != list(NAMES) or model.get("intercept") != 0.
            or any(model["standardizer"]["mean"])):
        raise ValueError("Ranker must preserve zero reference potential")
    return a.linear_scores(model, x)


def proposals(forks, ctx, model, cfg):
    best, tied, count = {}, set(), 0
    for batch in batches(transactions(forks, ctx), cfg["feature_batch_size"]):
        values = scores(model, features(batch, ctx))
        count += len(batch)
        for tx, value in zip(batch, values):
            if value <= 0:
                continue
            tx["gain"] = float(value)
            m, old = tx["mother"], best.get(tx["mother"])
            if old is None or value > old["gain"]+1e-10:
                best[m] = tx
                tied.discard(m)
            elif abs(value-old["gain"]) <= 1e-10:
                tied.add(m)
    return [best[m] for m in sorted(best) if m not in tied], dict(
        considered_alternatives=count, positive_best_mothers=len(best), tied_mothers=len(tied),
        uses_gt_for_edits=False, frozen_reference_features=True)
