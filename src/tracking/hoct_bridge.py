"""Explicit HOCT tensor adapter: five-frame, spatially bounded attention.

No calls to the unfinished create_graph_from_points(), automatic model downloads,
upstream solver, or distance-only relinking. All incoming alternatives for a
target share one context and the same orphan-aware parental softmax.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

from src.tracking.contracts import verify_file


def context_batches(coords, pairs, scale, config):
    """Core target tiles own edges; full parents plus a spatial halo provide context.

    Recursive subdivision never truncates candidate lists. If even one target's
    neighbourhood exceeds the cap, fail with a resource error instead of losing edges.
    """
    physical = coords[:, 1:] * np.asarray(scale)
    target_time = coords[pairs[:, 1], 0].astype(int)
    largest_edge = np.linalg.norm(physical[pairs[:, 1]] - physical[pairs[:, 0]], axis=1).max(initial=0)
    halo = max(config["halo_um"], float(largest_edge) + 1e-6)
    time_max = int(coords[:, 0].max())
    for t in np.unique(target_time):
        begin = max(0, min(int(t) - 2, time_max - 4))
        time_mask = (coords[:, 0] >= begin) & (coords[:, 0] < begin + 5)
        targets = np.unique(pairs[target_time == t, 1])
        tile_ids = np.floor(physical[targets] / config["tile_um"]).astype(int)
        groups = {}
        for target, tile in zip(targets, tile_ids):
            groups.setdefault(tuple(tile), []).append(int(target))
        pending = [np.asarray(group, int) for _, group in sorted(groups.items())]
        while pending:
            core = pending.pop()
            low, high = physical[core].min(axis=0) - halo, physical[core].max(axis=0) + halo
            node_ids = np.flatnonzero(time_mask & (physical >= low).all(axis=1) & (physical <= high).all(axis=1))
            included = np.zeros(len(coords), bool)
            included[node_ids] = True
            edge_ids = np.flatnonzero(included[pairs[:, 0]] & included[pairs[:, 1]])
            if len(node_ids) > config["max_nodes_per_batch"] or len(edge_ids) > config["max_edges_per_batch"]:
                if len(core) <= 1:
                    raise MemoryError("One target's complete HOCT context exceeds cap; adjust the declared resource budget")
                axis = int(np.ptp(physical[core], axis=0).argmax())
                ordered = core[np.argsort(physical[core, axis], kind="stable")]
                mid = len(ordered) // 2
                pending.extend([ordered[:mid], ordered[mid:]])
                continue
            owned = np.isin(pairs[edge_ids, 1], core)
            expected = np.flatnonzero(np.isin(pairs[:, 1], core))
            if not np.array_equal(edge_ids[owned], expected):
                raise ValueError("Tiling lost an incoming edge at a boundary")
            yield node_ids, edge_ids, owned


def score_edges(coords, pairs, features, scale, model_spec, config, device="cuda"):
    if not len(pairs):
        return np.empty(0, dtype=float)
    import torch
    # This private schema is checked against a source fingerprint in the manifest.
    # Only use a locally installed, audited HOCT revision and a local trusted model.
    import hoct._api as api
    verify_file(api.__file__, model_spec["api_sha256"])
    if model_spec.get("admissibility_reviewed") is not True or not model_spec.get("source_url"):
        raise ValueError("External HOCT weights require an explicit competition admissibility review")
    verify_file(model_spec["checkpoint"], model_spec["sha256"])
    mean, std = np.asarray(api._MEAN), np.asarray(api._STD)
    if features.shape != (len(coords), 19) or mean.shape != (19,) or std.shape != (19,) or (std <= 0).any():
        raise ValueError("Unsupported HOCT feature schema; do not guess normalization")
    model = torch.jit.load(model_spec["checkpoint"], map_location=device).eval()
    edge_logits = np.full(len(pairs), np.nan)
    orphan_logits = np.full(len(coords), np.nan)
    covered = np.zeros(len(pairs), dtype=np.int32)
    with torch.inference_mode():
        for nodes, edges, owned in context_batches(coords, pairs, scale, config):
            remap = np.full(len(coords), -1, dtype=np.int64)
            remap[nodes] = np.arange(len(nodes))
            local_pairs = remap[pairs[edges]]
            def tensor(array, dtype=torch.float32):
                return torch.as_tensor(array, dtype=dtype, device=device).unsqueeze(0)
            positions = coords[nodes, 1:]
            result = model(
                tensor((features[nodes] - mean) / std), tensor(positions),
                tensor((positions[local_pairs[:, 0]] + positions[local_pairs[:, 1]]) * .5),
                tensor(local_pairs, torch.int64), tensor(np.ones(len(nodes)), torch.bool),
                tensor(np.ones(len(edges)), torch.bool),
            )
            if len(result) != 4:
                raise ValueError("Unsupported HOCT model output ABI")
            logits = result[0].float().cpu().numpy().reshape(-1)
            orphans = result[3].float().cpu().numpy().reshape(-1)
            if logits.shape != (len(edges),) or orphans.shape != (len(nodes),):
                raise ValueError("Expected one edge logit and one orphan logit per item")
            if not np.isfinite(logits).all() or not np.isfinite(orphans).all():
                raise ValueError("HOCT produced non-finite logits")
            owned_edges = edges[owned]
            edge_logits[owned_edges] = logits[owned]
            covered[owned_edges] += 1
            targets = np.unique(pairs[owned_edges, 1])
            orphan_logits[targets] = orphans[remap[targets]]
    if not (covered == 1).all():
        raise ValueError("Every candidate must be scored exactly once, including tile boundaries")
    probabilities = np.empty(len(pairs), float)
    # Segment by child; softmax over parents, NOT over possible daughters.
    order = np.argsort(pairs[:, 1], kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(pairs[order, 1])) + 1, len(order)]
    for begin, end in zip(cuts[:-1], cuts[1:]):
        ids = order[begin:end]
        target = pairs[ids[0], 1]
        denominator = logsumexp(np.r_[edge_logits[ids], orphan_logits[target]])
        probabilities[ids] = np.exp(edge_logits[ids] - denominator)
    del model
    return probabilities


def blend_probabilities(hoct_prob, neural_prob, weight):
    if not 0 <= weight <= 1:
        raise ValueError("HOCT mixing weight must be between zero and one")
    # Unknown priors stay unknown: geometric proposals use HOCT evidence alone.
    return np.where(np.isfinite(neural_prob), weight * hoct_prob + (1 - weight) * neural_prob, hoct_prob)
