"""013 label-free scoring of the complete frozen hypothesis universe.

No edits are applied here. Keep all scores, including negative scores and intact
forks, so a later diagnostic cannot mistake abstention for good discrimination.
"""
from collections import defaultdict

import numpy as np

from core.nextgen import association as a, repair_ranker as r


def hypothesis_ids(forks, ctx):
    forks = np.asarray(forks, dtype=np.int64).reshape(-1, 3)
    prune = [(m, d, -1) for m, ds in sorted(ctx["children"].items())
             if len(ds) == 2 for d in sorted(ds)]
    ids = np.concatenate([forks, np.asarray(prune, np.int64).reshape(-1, 3)])
    if len(set(map(tuple, ids))) != len(ids):
        raise ValueError("Duplicate hypotheses; no tie-count inflation")
    return ids


def transaction(ids, ctx):
    m, x, y = map(int, ids)
    zeros = defaultdict(float)
    return a.transaction(m, [x] if y == -1 else [x, y], ctx, zeros, zeros,
                         {"orphan_penalty": 0.})


def reference(mother):
    # An identity transaction needs no fabricated edge or daughter ID.
    return dict(mother=int(mother), add=[], remove=[], affected_parents=[int(mother)])


def score_universe(forks, ctx, models, batch_size):
    if not isinstance(batch_size, int) or batch_size < 1 or not models:
        raise ValueError("Positive batch size and at least one model required")
    ids = hypothesis_ids(forks, ctx)
    values = np.empty((len(ids), len(models)), float)
    changed = np.empty(len(ids), bool)
    for start in range(0, len(ids), batch_size):
        rows = [transaction(i, ctx) for i in ids[start:start+batch_size]]
        features = r.features(rows, ctx)
        changed[start:start+len(rows)] = [bool(t["add"] or t["remove"]) for t in rows]
        for col, model in enumerate(models):
            values[start:start+len(rows), col] = r.scores(model, features)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite hypothesis scores")
    return ids, values, changed


def best_positive(ids, scores, changed):
    """Indices of unique positive best alternatives; ties abstain, no top-N."""
    ids, scores, changed = np.asarray(ids), np.asarray(scores), np.asarray(changed)
    if (ids.shape != (len(scores), 3) or changed.shape != scores.shape
            or changed.dtype != bool or not np.isfinite(scores).all()):
        raise ValueError("Invalid frozen score table")
    best, runner_up = {}, {}
    for i in np.flatnonzero(changed & (scores > 0)):
        m, old = int(ids[i, 0]), best.get(int(ids[i, 0]))
        if old is None:
            best[m] = int(i)
            runner_up[m] = -np.inf
        elif scores[i] > scores[old]:
            runner_up[m] = scores[old]
            best[m] = int(i)
        else:
            runner_up[m] = max(runner_up[m], scores[i])
    return np.asarray([best[m] for m in sorted(best)
                       if scores[best[m]]-runner_up[m] > 1e-10], np.int64)
