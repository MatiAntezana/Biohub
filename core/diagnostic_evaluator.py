"""
Diagnostic Evaluator for Autoresearch V2.
Implements the exact official metric formula of the Kaggle Cell Tracking competition
with deep error decomposition (node recall, penalty, edge jaccard, division jaccard).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from core.data_types import DiagnosticTelemetry


DEFAULT_SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (Z, Y, X) in um


def point_distance_um(p1: np.ndarray, p2: np.ndarray, scale: np.ndarray = DEFAULT_SCALE) -> float:
    """Calculate anisotropic physical distance in micrometers between two 3D coordinates."""
    diff = (np.asarray(p1) - np.asarray(p2)) * scale
    return float(np.sqrt(np.sum(diff ** 2)))


def match_nodes_bipartite(
    pred_nodes: List[Dict[str, Any]],
    gt_nodes: List[Dict[str, Any]],
    scale: np.ndarray = DEFAULT_SCALE,
    max_dist_um: float = 7.0,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Performs optimal bipartite matching between predicted and ground-truth nodes
    for a single timepoint within a physical distance threshold.
    Returns: (pred_to_gt, gt_to_pred) mappings.
    """
    if not pred_nodes or not gt_nodes:
        return {}, {}

    n_p = len(pred_nodes)
    n_g = len(gt_nodes)

    p_pos = np.array([n["pos"] for n in pred_nodes], dtype=np.float64) * scale
    g_pos = np.array([n["pos"] for n in gt_nodes], dtype=np.float64) * scale

    # Pairwise Euclidean distance
    dist_mat = np.sqrt(((p_pos[:, None] - g_pos[None, :]) ** 2).sum(axis=2))

    row_ind, col_ind = linear_sum_assignment(dist_mat)
    pred_to_gt = {}
    gt_to_pred = {}

    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] <= max_dist_um:
            pid = pred_nodes[r]["id"]
            gid = gt_nodes[c]["id"]
            pred_to_gt[pid] = gid
            gt_to_pred[gid] = pid

    return pred_to_gt, gt_to_pred


def compute_edge_confusion(
    pred_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
    pred_to_gt: Dict[int, int],
    gt_to_pred: Dict[int, int],
) -> Tuple[int, int, int]:
    """
    Computes (TP, FP, FN) for edges according to the official sparse ground-truth metric.
    """
    gt_edge_set: Set[Tuple[int, int]] = set(gt_edges)
    gt_source_map: Dict[int, Set[int]] = {}
    gt_target_map: Dict[int, Set[int]] = {}

    for u, v in gt_edges:
        gt_source_map.setdefault(u, set()).add(v)
        gt_target_map.setdefault(v, set()).add(u)

    tp = 0
    fp = 0
    accounted_gt_edges: Set[Tuple[int, int]] = set()

    for p_src, p_dst in pred_edges:
        g_src = pred_to_gt.get(p_src)
        g_dst = pred_to_gt.get(p_dst)

        if g_src is not None and g_dst is not None:
            if (g_src, g_dst) in gt_edge_set:
                tp += 1
                accounted_gt_edges.add((g_src, g_dst))
                continue

        # Check for false positive: connects to a GT node that in GT has a different edge
        is_fp = False
        if g_src is not None:
            # Source matches a GT node with different targets in GT
            if g_src in gt_source_map and (g_dst is None or g_dst not in gt_source_map[g_src]):
                is_fp = True
        if g_dst is not None:
            # Target matches a GT node with different sources in GT
            if g_dst in gt_target_map and (g_src is None or g_src not in gt_target_map[g_dst]):
                is_fp = True

        if is_fp:
            fp += 1

    fn = len(gt_edge_set - accounted_gt_edges)
    return tp, fp, fn


def compute_division_confusion(
    pred_nodes_by_id: Dict[int, Dict[str, Any]],
    pred_edges: List[Tuple[int, int]],
    gt_nodes_by_id: Dict[int, Dict[str, Any]],
    gt_edges: List[Tuple[int, int]],
    pred_to_gt: Dict[int, int],
    gt_to_pred: Dict[int, int],
) -> Tuple[int, int, int]:
    """
    Computes (div_tp, div_fp, div_fn) for mitotic cell divisions.
    A division in GT is a node with out-degree >= 2.
    Evaluated with bipartite assignment in local window t +/- 1.
    """
    gt_out: Dict[int, List[int]] = {}
    for u, v in gt_edges:
        gt_out.setdefault(u, []).append(v)
    gt_divisions = [u for u, targets in gt_out.items() if len(targets) >= 2]

    pred_out: Dict[int, List[int]] = {}
    for u, v in pred_edges:
        pred_out.setdefault(u, []).append(v)
    pred_forks = [u for u, targets in pred_out.items() if len(targets) >= 2]

    if not gt_divisions:
        return 0, len(pred_forks), 0
    if not pred_forks:
        return 0, 0, len(gt_divisions)

    # Cost matrix between candidate predicted forks and GT divisions
    cost_matrix = np.full((len(pred_forks), len(gt_divisions)), 1e6)

    for p_idx, p_fork in enumerate(pred_forks):
        p_node = pred_nodes_by_id.get(p_fork)
        if not p_node:
            continue
        p_targets = pred_out.get(p_fork, [])

        for g_idx, g_div in enumerate(gt_divisions):
            g_node = gt_nodes_by_id.get(g_div)
            if not g_node:
                continue
            g_targets = gt_out.get(g_div, [])

            # Check timepoint window |t_p - t_g| <= 1
            if abs(p_node["t"] - g_node["t"]) > 1:
                continue

            # Check if parent matches or is close
            matched_g_parent = pred_to_gt.get(p_fork)
            parent_dist = point_distance_um(p_node["pos"], g_node["pos"])
            if matched_g_parent != g_div and parent_dist > 7.0:
                continue

            # Check matching of daughter branches
            matched_daughters = 0
            for pt in p_targets:
                gt_match = pred_to_gt.get(pt)
                if gt_match is not None and gt_match in g_targets:
                    matched_daughters += 1

            if matched_daughters >= 2 or (matched_daughters >= 1 and parent_dist <= 5.0):
                cost_matrix[p_idx, g_idx] = parent_dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    div_tp = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e5:
            div_tp += 1

    div_fp = max(0, len(pred_forks) - div_tp)
    div_fn = max(0, len(gt_divisions) - div_tp)
    return div_tp, div_fp, div_fn


def decompose_error_mechanisms(
    pred_nodes_by_id: Dict[int, Dict[str, Any]],
    gt_nodes_by_id: Dict[int, Dict[str, Any]],
    pred_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
    pred_to_gt: Dict[int, int],
    gt_to_pred: Dict[int, int],
) -> Dict[str, int]:
    """
    Decomposes why edges and nodes were lost into 4 mechanistic root causes:
    - recovered: successfully linked TP edges
    - lost_to_detection: GT edge failed because endpoints were not detected
    - fragmented: endpoints detected, but tracker failed to link them
    - wrong_association: tracker linked to the wrong cell
    """
    recovered = 0
    lost_to_detection = 0
    fragmented = 0
    pred_edge_set = set(pred_edges)

    for g_u, g_v in gt_edges:
        p_u = gt_to_pred.get(g_u)
        p_v = gt_to_pred.get(g_v)

        if p_u is None or p_v is None:
            lost_to_detection += 1
        elif (p_u, p_v) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1

    wrong_assoc = 0
    for p_u, p_v in pred_edges:
        g_u = pred_to_gt.get(p_u)
        g_v = pred_to_gt.get(p_v)
        if g_u is not None and g_v is not None and (g_u, g_v) not in gt_edges:
            wrong_assoc += 1

    return {
        "recovered": recovered,
        "lost_to_detection": lost_to_detection,
        "fragmented": fragmented,
        "wrong_association": wrong_assoc,
    }


def evaluate_tracking(
    pred_nodes_by_t: Dict[int, List[Dict[str, Any]]],
    pred_edges: List[Tuple[int, int]],
    gt_nodes_by_t: Dict[int, List[Dict[str, Any]]],
    gt_edges: List[Tuple[int, int]],
    estimated_total_nodes: Optional[int] = None,
    scale: np.ndarray = DEFAULT_SCALE,
) -> DiagnosticTelemetry:
    """
    Full diagnostic evaluation of predicted tracking against ground truth.
    Returns a DiagnosticTelemetry dataclass with all metrics and sub-metrics.
    """
    all_timepoints = sorted(set(pred_nodes_by_t.keys()).union(gt_nodes_by_t.keys()))

    global_pred_to_gt: Dict[int, int] = {}
    global_gt_to_pred: Dict[int, int] = {}
    total_matched_gt_nodes = 0
    total_gt_nodes = sum(len(nodes) for nodes in gt_nodes_by_t.values())
    total_pred_nodes = sum(len(nodes) for nodes in pred_nodes_by_t.values())

    pred_nodes_by_id: Dict[int, Dict[str, Any]] = {}
    gt_nodes_by_id: Dict[int, Dict[str, Any]] = {}

    for t in all_timepoints:
        p_nodes = pred_nodes_by_t.get(t, [])
        g_nodes = gt_nodes_by_t.get(t, [])

        for n in p_nodes:
            pred_nodes_by_id[n["id"]] = n
        for n in g_nodes:
            gt_nodes_by_id[n["id"]] = n

        p2g, g2p = match_nodes_bipartite(p_nodes, g_nodes, scale=scale, max_dist_um=7.0)
        global_pred_to_gt.update(p2g)
        global_gt_to_pred.update(g2p)
        total_matched_gt_nodes += len(g2p)

    node_recall = float(total_matched_gt_nodes / max(1, total_gt_nodes))
    spurious_nodes = max(0, total_pred_nodes - len(global_pred_to_gt))
    missed_gt_nodes = max(0, total_gt_nodes - total_matched_gt_nodes)

    # Edge confusion
    edge_tp, edge_fp, edge_fn = compute_edge_confusion(
        pred_edges, gt_edges, global_pred_to_gt, global_gt_to_pred
    )
    unadjusted_edge_jaccard = float(edge_tp / max(1, edge_tp + edge_fp + edge_fn))

    # Node budget penalty calculation
    # If estimated_total_nodes is not provided, default to total_gt_nodes
    t_true = estimated_total_nodes if estimated_total_nodes and estimated_total_nodes > 0 else max(1, total_gt_nodes)
    if total_pred_nodes > t_true:
        penalty_ratio = (total_pred_nodes - t_true) / float(t_true)
        node_penalty_multiplier = max(0.0, 1.0 - 0.1 * penalty_ratio)
    else:
        node_penalty_multiplier = 1.0

    adjusted_edge_jaccard = unadjusted_edge_jaccard * node_penalty_multiplier

    # Division confusion
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes_by_id, pred_edges, gt_nodes_by_id, gt_edges, global_pred_to_gt, global_gt_to_pred
    )
    div_denom = div_tp + div_fp + div_fn
    division_jaccard = float(div_tp / div_denom) if div_denom > 0 else 0.0

    # Total official competition score
    total_score = adjusted_edge_jaccard + 0.1 * division_jaccard

    # Error decomposition
    decomp = decompose_error_mechanisms(
        pred_nodes_by_id, gt_nodes_by_id, pred_edges, gt_edges, global_pred_to_gt, global_gt_to_pred
    )

    # Displacement errors
    z_diffs = []
    xy_diffs = []
    for pid, gid in global_pred_to_gt.items():
        p_pos = np.asarray(pred_nodes_by_id[pid]["pos"], dtype=float)
        g_pos = np.asarray(gt_nodes_by_id[gid]["pos"], dtype=float)
        z_diffs.append(abs(p_pos[0] - g_pos[0]) * scale[0])
        xy_dist = float(np.sqrt(np.sum(((p_pos[1:] - g_pos[1:]) * scale[1:]) ** 2)))
        xy_diffs.append(xy_dist)

    z_err = float(np.mean(z_diffs)) if z_diffs else 0.0
    xy_err = float(np.mean(xy_diffs)) if xy_diffs else 0.0

    return DiagnosticTelemetry(
        total_score=round(total_score, 4),
        adjusted_edge_jaccard=round(adjusted_edge_jaccard, 4),
        division_jaccard=round(division_jaccard, 4),
        unadjusted_edge_jaccard=round(unadjusted_edge_jaccard, 4),
        node_recall_7um=round(node_recall, 4),
        total_predicted_nodes=total_pred_nodes,
        estimated_true_nodes=t_true,
        node_penalty_multiplier=round(node_penalty_multiplier, 4),
        spurious_nodes=spurious_nodes,
        missed_gt_nodes=missed_gt_nodes,
        div_tp=div_tp,
        div_fp=div_fp,
        div_fn=div_fn,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        edges_recovered=decomp["recovered"],
        edges_fragmented=decomp["fragmented"],
        edges_lost_to_detection=decomp["lost_to_detection"],
        wrong_association_edges=decomp["wrong_association"],
        z_axis_displacement_error=round(z_err, 3),
        xy_displacement_error=round(xy_err, 3),
    )
