"""Optional one-frame rescue at mature, mutually isolated track interruptions."""
from __future__ import annotations
from collections import defaultdict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.nextgen.images import normalized_correlation


def propose_rescues(lineage, source, config):
    c, scale = lineage.coords, source.scale
    children, parent = defaultdict(list), {}
    for u, v in lineage.edges:
        children[int(u)].append(int(v))
        parent[int(v)] = int(u)
    # Exclude one-frame tracks, mitotic endpoints, and ambiguous candidate pairs.
    ends = [i for i in range(len(c)) if not children[i] and i in parent and len(children[parent[i]]) == 1]
    starts = [i for i in range(len(c)) if i not in parent and len(children[i]) == 1]
    accepted, proposed_positions = [], defaultdict(list)
    for t in sorted({int(c[i, 0]) for i in ends}):
        left = np.asarray([i for i in ends if c[i, 0] == t], int)
        right = np.asarray([i for i in starts if c[i, 0] == t + 2], int)
        if not len(right):
            continue
        lt, rt = cKDTree(c[left, 1:] * scale), cKDTree(c[right, 1:] * scale)
        forward = rt.query_ball_point(c[left, 1:] * scale, 2 * config["step_radius_um"])
        backward = lt.query_ball_point(c[right, 1:] * scale, 2 * config["step_radius_um"])
        present = c[c[:, 0] == t + 1, 1:] * scale
        occupied = cKDTree(present) if len(present) else None
        for local_u, hits in enumerate(forward):
            if len(hits) != 1 or backward[hits[0]] != [local_u]:
                continue
            u, v = int(left[local_u]), int(right[hits[0]])
            midpoint = (c[u, 1:] + c[v, 1:]) * .5
            half = np.ceil(config["search_radius_um"] / scale).astype(int)
            origin = np.maximum(0, np.floor(midpoint).astype(int) - half)
            stop = np.minimum(source.shape[1:], np.ceil(midpoint).astype(int) + half + 1)
            crop = source.frame(t + 1)[tuple(slice(a, b) for a, b in zip(origin, stop))]
            smooth = ndimage.gaussian_filter(crop, sigma=config["peak_sigma_um"] / scale)
            grid = np.indices(crop.shape).reshape(3, -1).T + origin
            inside = np.linalg.norm((grid - midpoint) * scale, axis=1) <= config["search_radius_um"]
            peak_ids = np.flatnonzero((smooth == ndimage.maximum_filter(smooth, size=3)).ravel() & inside)
            # Two comparable maxima are an ambiguous rescue, not two new cells.
            ranked = sorted(peak_ids, key=lambda i: (-smooth.ravel()[i], int(i)))
            if not ranked:
                continue
            point = grid[ranked[0]].astype(float)
            distant = [i for i in ranked[1:] if np.linalg.norm((grid[i] - point) * scale) > config["exclusion_um"]]
            if distant and smooth.ravel()[distant[0]] >= config["ambiguity_ratio"] * smooth.ravel()[ranked[0]]:
                continue
            if occupied is not None and occupied.query(point * scale)[0] < config["exclusion_um"]:
                continue
            if any(np.linalg.norm((point - q) * scale) < config["exclusion_um"] for q in proposed_positions[t + 1]):
                continue
            if max(np.linalg.norm((point - c[i, 1:]) * scale) for i in (u, v)) > config["step_radius_um"]:
                continue
            centre_patch = source.patch(t + 1, point, half_um=4.0)
            background = np.median(centre_patch)
            mad = np.median(np.abs(centre_patch - background))
            contrast = float((source.frame(t + 1)[tuple(point.astype(int))] - background) / max(1.4826 * mad, 1e-4))
            agreement = min(normalized_correlation(centre_patch, source.patch(int(c[i, 0]), c[i, 1:], half_um=4.0)) for i in (u, v))
            if contrast < config["min_snr"] or agreement < config["min_ncc"]:
                continue
            accepted.append({"source": u, "target": v, "coord": [t + 1, *point.tolist()],
                             "snr": contrast, "ncc": agreement})
            proposed_positions[t + 1].append(point)
    maximum = min(config["max_nodes"], int(len(c) * config["max_fraction"]))
    # Rank only image-supported proposals. This cap limits growth; it does not pad a CSV.
    accepted.sort(key=lambda r: (-r["ncc"], -r["snr"], r["source"], r["target"]))
    return accepted[:maximum]
