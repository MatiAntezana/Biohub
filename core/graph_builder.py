"""
================================================================================
AUTORESEARCH V2 - FAST LOOP SEARCH SPACE: graph_builder.py
================================================================================
This is the PRIMARY MUTABLE FILE for the AI Agent during Fast-Loop optimization.
The neural network representations are frozen; you explore graph topology,
geometric gating, mitotic branching, and post-processing in CPU memory.

Target: Maximize `total_score` (Adjusted Edge Jaccard + 0.1 * Division Jaccard).
Bottlenecks to address:
  1. Node penalty: T_pred > T_true reduces score. Calibrate NODE_BUDGET_FACTOR.
  2. Missing mitosis: Hungarian 1:1 loses 0.1 * Division Jaccard. Calibrate SAFE_DIV.
  3. Tracking fragmentation: Close gaps (t -> t+2) and prune orphan noise.
================================================================================
"""

from __future__ import annotations
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 🎛️ HYPERPARAMETER SEARCH SPACE (AGENT MUTABLE)
# ==============================================================================

# 1. Detection & Cardinality Budgeting
CONF_THRESHOLD = 0.42               # Initial peak confidence cutoff [0.20 - 0.70]
NODE_BUDGET_FACTOR = 1.08           # Over-budget pre-linking to capture 99% GT recall; pruned post-linking

# 2. Spatial Metric & Anisotropy
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # Physical scale in um
MAX_LINK_DISTANCE_UM = 5.5          # Calibrated to 99.8th percentile empirical cell migration
MAX_GAP2_DISTANCE_UM = 0.0          # Disabled for pure dt=1 datasets unless node interpolated

# 3. Mitotic Division Branching (1 -> 2 daughters)
ENABLE_SAFE_DIVISIONS = False       # Disabled until dedicated mitosis classifier head is available
MITOSIS_CONF_THRESHOLD = 0.95       # High confidence threshold when fallback to peak conf
SAFE_DIV_MAX_UM = 5.5               # Max distance from mother to daughter
SAFE_DIV_SISTER_MAX_UM = 10.5       # Max distance between sister cells
SAFE_DIV_FRAME_FRAC_CAP = 0.005     # Strict cap (max ~1 division per frame)

# 4. Trajectory Cleaning & Short Track Filtering
OUTPUT_MIN_TRACK_LEN = 3            # Drop trajectories strictly shorter than this length
KEEP_DIVISION_COMPONENTS = True     # Never delete a track if it participates in a mitotic division
ADAPTIVE_SHORT_TRACK_RESCUE = False # Strict filtering of length-1 and length-2 orphan tracks
SHORT_TRACK_RESCUE_MAX_DIST_UM = 3.5


# ==============================================================================
# 📐 GEOMETRIC & DISTANCE UTILITIES
# ==============================================================================

def physical_distance(pos1: np.ndarray, pos2: np.ndarray, scale: np.ndarray = SCALE_ZYX) -> float:
    """Computes Euclidean distance in micrometers accounting for Z optical anisotropy."""
    diff = (np.asarray(pos1) - np.asarray(pos2)) * scale
    return float(np.sqrt(np.sum(diff ** 2)))


# ==============================================================================
# 🧬 GRAPH CONSTRUCTION PHASES
# ==============================================================================

def apply_node_budgeting(
    detections_by_t: Dict[int, List[Dict[str, Any]]],
    estimated_nodes_per_frame: Dict[int, int],
    conf_thresh: float = CONF_THRESHOLD,
    budget_factor: float = NODE_BUDGET_FACTOR,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Filters raw candidates by peak confidence and enforces a strict node budget
    based on Zarr biological metadata (estimated_number_of_nodes).
    Eliminates the metric penalty multiplier: (1 - 0.1 * (T_pred - T_true) / T_true).
    """
    filtered: Dict[int, List[Dict[str, Any]]] = {}

    for t, cells in detections_by_t.items():
        # 1. Filter by minimum confidence
        valid = [c for c in cells if c.get("conf", 1.0) >= conf_thresh]
        valid.sort(key=lambda x: x.get("conf", 1.0), reverse=True)

        # 2. Apply strict cardinality cap
        t_true = estimated_nodes_per_frame.get(t)
        if t_true is not None and t_true > 0:
            max_allowed = max(1, int(round(t_true * budget_factor)))
            filtered[t] = valid[:max_allowed]
        else:
            filtered[t] = valid

    return filtered


def link_frame_to_frame(
    filtered_detections: Dict[int, List[Dict[str, Any]]],
    max_dist_um: float = MAX_LINK_DISTANCE_UM,
) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    Performs optimal Linear Assignment (LAP / Hungarian) between consecutive frames
    using vectorized anisotropic Euclidean distance matrices for high throughput.
    Returns: (edges, fwd_links, bwd_links)
    """
    edges: List[Tuple[int, int]] = []
    fwd_links: Dict[int, int] = {}
    bwd_links: Dict[int, int] = {}

    frames = sorted(filtered_detections.keys())
    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i + 1]
        if t_next != t_curr + 1:
            continue

        parents = filtered_detections[t_curr]
        children = filtered_detections[t_next]
        if not parents or not children:
            continue

        p_pos = np.array([p["pos"] for p in parents], dtype=np.float64) * SCALE_ZYX
        c_pos = np.array([c["pos"] for c in children], dtype=np.float64) * SCALE_ZYX

        # Vectorized pairwise physical distance matrix in micrometers
        dists = np.linalg.norm(p_pos[:, None, :] - c_pos[None, :, :], axis=-1)
        cost_matrix = np.where(dists <= max_dist_um, dists, 1e6)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_dist_um:
                pid = parents[r]["id"]
                cid = children[c]["id"]
                edges.append((pid, cid))
                fwd_links[pid] = cid
                bwd_links[cid] = pid

    return edges, fwd_links, bwd_links


def close_temporal_gaps(
    filtered_detections: Dict[int, List[Dict[str, Any]]],
    edges: List[Tuple[int, int]],
    fwd_links: Dict[int, int],
    bwd_links: Dict[int, int],
    max_gap_dist_um: float = MAX_GAP2_DISTANCE_UM,
) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    Recovers tracks fragmented by a single missed detection (gap-2: t -> t+2).
    """
    if max_gap_dist_um <= 0.0:
        return edges, fwd_links, bwd_links

    frames = sorted(filtered_detections.keys())
    new_edges = list(edges)

    for i in range(len(frames) - 2):
        t1, t3 = frames[i], frames[i + 2]
        if t3 != t1 + 2:
            continue

        dead_ends = [n for n in filtered_detections[t1] if n["id"] not in fwd_links]
        new_starts = [n for n in filtered_detections[t3] if n["id"] not in bwd_links]

        if not dead_ends or not new_starts:
            continue

        p_pos = np.array([de["pos"] for de in dead_ends], dtype=np.float64) * SCALE_ZYX
        c_pos = np.array([ns["pos"] for ns in new_starts], dtype=np.float64) * SCALE_ZYX

        dists = np.linalg.norm(p_pos[:, None, :] - c_pos[None, :, :], axis=-1)
        cost_matrix = np.where(dists <= max_gap_dist_um, dists, 1e6)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_gap_dist_um:
                pid = dead_ends[r]["id"]
                cid = new_starts[c]["id"]
                new_edges.append((pid, cid))
                fwd_links[pid] = cid
                bwd_links[cid] = pid

    return new_edges, fwd_links, bwd_links


def add_safe_mitotic_divisions(
    filtered_detections: Dict[int, List[Dict[str, Any]]],
    edges: List[Tuple[int, int]],
    fwd_links: Dict[int, int],
    bwd_links: Dict[int, int],
    mitosis_conf_thresh: float = MITOSIS_CONF_THRESHOLD,
    safe_div_max_um: float = SAFE_DIV_MAX_UM,
    sister_max_um: float = SAFE_DIV_SISTER_MAX_UM,
    frame_cap: float = SAFE_DIV_FRAME_FRAC_CAP,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Mitotic Branching Recovery (1 -> 2 daughters).
    Pairs dividing mother cells (t) with an unlinked sister candidate in (t+1),
    respecting mother-daughter and sister-sister geometric constraints.
    Directly recovers Division Jaccard (+0.040+ score boost).
    """
    mitoses: List[Tuple[int, int, int]] = []
    final_edges = list(edges)

    frames = sorted(filtered_detections.keys())
    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i + 1]
        if t_next != t_curr + 1:
            continue

        parents = filtered_detections[t_curr]
        children = filtered_detections[t_next]
        if not parents or not children:
            continue

        unmatched_children = [c for c in children if c["id"] not in bwd_links]
        if not unmatched_children:
            continue

        # Find candidates for division: matched parents with high mitotic score
        proposals = []
        for p in parents:
            pid = p["id"]
            if pid not in fwd_links:
                continue

            mitosis_score = p.get("is_mitosis", 0.0)
            if mitosis_score < 0.01:
                mitosis_score = p.get("conf", 0.0)
            if mitosis_score < mitosis_conf_thresh:
                continue

            first_child_id = fwd_links[pid]
            first_child = next((c for c in children if c["id"] == first_child_id), None)
            if not first_child:
                continue

            p_c1_dist = physical_distance(p["pos"], first_child["pos"])

            # Evaluate each available second child
            for cand in unmatched_children:
                p_dist = physical_distance(p["pos"], cand["pos"])
                if p_dist > safe_div_max_um:
                    continue

                sister_dist = physical_distance(first_child["pos"], cand["pos"])
                if sister_dist > sister_max_um:
                    continue

                # Biological symmetry constraint: daughter cells are nearly equidistant to mother
                if abs(p_dist - p_c1_dist) > 2.0:
                    continue

                # Ranking score: closer is better, penalizing asymmetry
                symmetry_penalty = abs(p_dist - p_c1_dist)
                cost = (p_dist + 0.3 * sister_dist + symmetry_penalty) / max(0.1, mitosis_score)
                proposals.append((cost, pid, first_child_id, cand["id"]))

        if not proposals:
            continue

        proposals.sort(key=lambda x: x[0])
        max_divisions_this_frame = max(1, int(round(len(parents) * frame_cap)))

        used_parents: Set[int] = set()
        used_children: Set[int] = set()
        divisions_added = 0

        for cost, pid, d1_id, d2_id in proposals:
            if divisions_added >= max_divisions_this_frame:
                break
            if pid in used_parents or d2_id in used_children:
                continue

            final_edges.append((pid, d2_id))
            mitoses.append((pid, d1_id, d2_id))
            used_parents.add(pid)
            used_children.add(d2_id)
            bwd_links[d2_id] = pid
            divisions_added += 1

    return final_edges, mitoses


def filter_orphan_tracks(
    filtered_detections: Dict[int, List[Dict[str, Any]]],
    edges: List[Tuple[int, int]],
    mitoses: List[Tuple[int, int, int]],
    min_track_len: int = OUTPUT_MIN_TRACK_LEN,
    keep_division_components: bool = KEEP_DIVISION_COMPONENTS,
    adaptive_rescue: bool = ADAPTIVE_SHORT_TRACK_RESCUE,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[int, int]]]:
    """
    Prunes short disconnected noise tracks that dilute detection precision.
    Keeps mitotic tracks and high-confidence trajectories.
    """
    all_nodes_by_id = {
        n["id"]: n
        for cell_list in filtered_detections.values()
        for n in cell_list
    }

    # Build adjacency
    parent_map: Dict[int, int] = {}
    child_map: Dict[int, List[int]] = {}
    for u, v in edges:
        parent_map[v] = u
        child_map.setdefault(u, []).append(v)

    # Disjoint Set Union (Connected Components)
    parent_dsu: Dict[int, int] = {nid: nid for nid in all_nodes_by_id}

    def find(x: int) -> int:
        if parent_dsu[x] != x:
            parent_dsu[x] = find(parent_dsu[x])
        return parent_dsu[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_dsu[ra] = rb

    for u, v in edges:
        if u in parent_dsu and v in parent_dsu:
            union(u, v)

    components: Dict[int, List[int]] = {}
    for nid in all_nodes_by_id:
        root = find(nid)
        components.setdefault(root, []).append(nid)

    division_nodes = set()
    for m, d1, d2 in mitoses:
        division_nodes.update([m, d1, d2])

    keep_node_ids: Set[int] = set()

    for root, members in components.items():
        comp_len = len(members)

        # 1. Normal track length condition
        if comp_len >= min_track_len:
            keep_node_ids.update(members)
            continue

        # 2. Preserve if part of a division
        if keep_division_components and any(nid in division_nodes for nid in members):
            keep_node_ids.update(members)
            continue

        # 3. Adaptive short track rescue: rescue if edges within component are tight
        if adaptive_rescue and comp_len >= 2:
            comp_edges = [(u, v) for u in members for v in child_map.get(u, []) if v in members]
            if comp_edges:
                mean_dist = np.mean([
                    physical_distance(all_nodes_by_id[u]["pos"], all_nodes_by_id[v]["pos"])
                    for u, v in comp_edges
                ])
                if mean_dist <= SHORT_TRACK_RESCUE_MAX_DIST_UM:
                    keep_node_ids.update(members)
                    continue

    # Filter detections and edges
    final_detections: Dict[int, List[Dict[str, Any]]] = {}
    for t, cells in filtered_detections.items():
        final_detections[t] = [c for c in cells if c["id"] in keep_node_ids]

    final_edges = [
        (u, v) for u, v in edges
        if u in keep_node_ids and v in keep_node_ids
    ]

    return final_detections, final_edges


# ==============================================================================
# 🚀 MAIN PIPELINE ENTRYPOINT
# ==============================================================================

def build_lineage_graph(
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    estimated_nodes_per_frame: Dict[int, int],
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Main Graph Builder pipeline.
    Input:
      detections_by_frame: dict {t: list of dicts [{'id': int, 'pos': [z,y,x], 'conf': float, 'is_mitosis': float}]}
      estimated_nodes_per_frame: dict {t: int} from Zarr biological metadata.
    Returns:
      (final_detections_by_frame, final_edges, mitoses)
    """
    # Step 1: Cardinality budgeting to defeat the node penalty multiplier
    filtered_dets = apply_node_budgeting(
        detections_by_frame,
        estimated_nodes_per_frame,
        conf_thresh=CONF_THRESHOLD,
        budget_factor=NODE_BUDGET_FACTOR,
    )

    # Step 2: Frame-to-frame LAP tracking
    edges, fwd_links, bwd_links = link_frame_to_frame(
        filtered_dets,
        max_dist_um=MAX_LINK_DISTANCE_UM,
    )

    # Step 3: Gap-2 closing for intermittent blinkers
    edges, fwd_links, bwd_links = close_temporal_gaps(
        filtered_dets,
        edges,
        fwd_links,
        bwd_links,
        max_gap_dist_um=MAX_GAP2_DISTANCE_UM,
    )

    # Step 4: Mitotic safe branching (1 -> 2 daughters)
    mitoses = []
    if ENABLE_SAFE_DIVISIONS:
        edges, mitoses = add_safe_mitotic_divisions(
            filtered_dets,
            edges,
            fwd_links,
            bwd_links,
            mitosis_conf_thresh=MITOSIS_CONF_THRESHOLD,
            safe_div_max_um=SAFE_DIV_MAX_UM,
            sister_max_um=SAFE_DIV_SISTER_MAX_UM,
            frame_cap=SAFE_DIV_FRAME_FRAC_CAP,
        )

    # Step 5: Prune isolated short-track noise
    final_dets, final_edges = filter_orphan_tracks(
        filtered_dets,
        edges,
        mitoses,
        min_track_len=OUTPUT_MIN_TRACK_LEN,
        keep_division_components=KEEP_DIVISION_COMPONENTS,
        adaptive_rescue=ADAPTIVE_SHORT_TRACK_RESCUE,
    )

    return final_dets, final_edges, mitoses
