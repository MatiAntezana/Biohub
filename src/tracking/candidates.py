"""Physical-space alternatives; distances propose edges, they do not decide them."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def mutual_mapping(anchor, raw, scale, tolerance_um):
    """Transfer neural priors only through unique mutual point matches.

    Rounded/smoothed reference coordinates are not raw detector array indices.
    Ambiguous matches receive no prior; they are never coerced into another ID.
    """
    mapping = {}
    for t in np.unique(anchor[:, 0]):
        a, r = np.flatnonzero(anchor[:, 0] == t), np.flatnonzero(raw[:, 0] == t)
        if not len(r):
            continue
        ap, rp = anchor[a, 1:] * scale, raw[r, 1:] * scale
        at, rt = cKDTree(ap), cKDTree(rp)
        neighbours = rt.query_ball_point(ap, tolerance_um)
        reverse = at.query_ball_point(rp, tolerance_um)
        for i, hits in enumerate(neighbours):
            if len(hits) == 1 and reverse[hits[0]] == [i]:
                mapping[int(r[hits[0]])] = int(a[i])
    return mapping


def build_candidates(anchor, raw_coords, raw_edges, scale, config):
    scale = np.asarray(scale)
    mapping = mutual_mapping(anchor.coords, raw_coords, scale, config["prior_match_um"])
    priors = {tuple(map(int, edge)): float("nan") for edge in anchor.edges}
    for u, v, probability, _ in raw_edges:
        u, v = mapping.get(int(u)), mapping.get(int(v))
        if u is not None and v is not None and anchor.coords[v, 0] == anchor.coords[u, 0] + 1:
            key = (u, v)
            old = priors.get(key, float("nan"))
            priors[key] = float(probability) if np.isnan(old) else max(old, float(probability))
    for t in np.unique(anchor.coords[:, 0]):
        source = np.flatnonzero(anchor.coords[:, 0] == t - 1)
        target = np.flatnonzero(anchor.coords[:, 0] == t)
        if not len(source):
            continue
        tree = cKDTree(anchor.coords[source, 1:] * scale)
        distances, indices = tree.query(anchor.coords[target, 1:] * scale,
            k=min(config["neighbours"], len(source)), distance_upper_bound=config["radius_um"])
        distances, indices = distances.reshape(len(target), -1), indices.reshape(len(target), -1)
        for v, ds, ids in zip(target, distances, indices):
            for d, index in zip(ds, ids):
                if np.isfinite(d):
                    priors.setdefault((int(source[index]), int(v)), float("nan"))
    pairs = np.asarray(sorted(priors), dtype=np.int64).reshape(-1, 2)
    probabilities = np.asarray([priors[tuple(edge)] for edge in pairs], dtype=float)
    if len(pairs) > config["max_edges"]:
        raise ValueError("Candidate graph exceeds the explicit memory budget")
    return pairs, probabilities, {"matched_prior_nodes": len(mapping), "anchor_nodes": len(anchor.coords),
        "candidate_edges": len(pairs), "edges_without_neural_prior": int(np.isnan(probabilities).sum())}
