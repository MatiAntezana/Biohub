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
CONF_THRESHOLD = 0.970              # Calibrated peak confidence cutoff to suppress background noise
NODE_BUDGET_FACTOR = 1.22           # Optimal budget factor balancing GT recall and node penalty multiplier

# 2. Spatial Metric & Anisotropy
SCALE_ZYX = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # Physical scale in um
MAX_LINK_DISTANCE_UM = 5.5          # Calibrated to 99.8th percentile empirical cell migration
MAX_GAP2_DISTANCE_UM = 0.0          # Disabled for pure dt=1 datasets unless node interpolated

# 2b. Neural Representation Edge Gating
NEURAL_EDGE_WEIGHT = 0.12           # Weight of neural candidate edge affinities in cost matrix
NEURAL_MAX_PROB_DIST_UM = 6.2       # Extended distance envelope allowed for high-probability neural edges
NEURAL_PROB_THRESHOLD = 0.20        # Minimum probability required to utilize extended distance envelope

# 3. Mitotic Division Branching (Cytokinesis Divergence Gating)
ENABLE_SAFE_DIVISIONS = True        # Enabled with strict forward temporal divergence
MITOSIS_MIN_DIST_UM = 3.5           # Min distance from mother to daughter
MITOSIS_MAX_DIST_UM = 6.0           # Max distance from mother to daughter
MITOSIS_SISTER_MIN_UM = 9.5         # Min distance between sister cells
MITOSIS_SISTER_MAX_UM = 11.5        # Max distance between sister cells
MITOSIS_SYM_DIFF_UM = 0.28          # Max asymmetry |d1 - d2| between daughters
MITOSIS_COS_THRESH = -0.95          # Collinear cleavage spindle (angle >= 162 deg)
MITOSIS_MIN_MOTHER_LEN = 5          # Interphase maturity: mother must track >= 5 frames
MITOSIS_MIN_PROB = 0.20             # Neural probability support for cleavage edge

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
        t_key = int(t) if isinstance(t, str) and t.isdigit() else t
        t_true = estimated_nodes_per_frame.get(t_key) or estimated_nodes_per_frame.get(str(t))
        if t_true is not None and t_true > 0:
            max_allowed = max(1, int(round(t_true * budget_factor)))
            filtered[t] = valid[:max_allowed]
        else:
            filtered[t] = valid

    return filtered


def link_frame_to_frame(
    filtered_detections: Dict[int, List[Dict[str, Any]]],
    max_dist_um: float = MAX_LINK_DISTANCE_UM,
    edge_candidates: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
    neural_weight: float = NEURAL_EDGE_WEIGHT,
    max_prob_dist_um: float = NEURAL_MAX_PROB_DIST_UM,
    prob_threshold: float = NEURAL_PROB_THRESHOLD,
) -> Tuple[List[Tuple[int, int]], Dict[int, int], Dict[int, int]]:
    """
    Performs optimal Linear Assignment (LAP / Hungarian) between consecutive frames
    using vectorized anisotropic Euclidean distance matrices blended with neural edge representation logits.
    Returns: (edges, fwd_links, bwd_links)
    """
    edges: List[Tuple[int, int]] = []
    fwd_links: Dict[int, int] = {}
    bwd_links: Dict[int, int] = {}

    # Sort frames numerically
    frames = sorted(filtered_detections.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x)
    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i + 1]
        try:
            if int(t_next) != int(t_curr) + 1:
                continue
        except (ValueError, TypeError):
            continue

        parents = filtered_detections[t_curr]
        children = filtered_detections[t_next]
        if not parents or not children:
            continue

        p_pos = np.array([p["pos"] for p in parents], dtype=np.float64) * SCALE_ZYX
        c_pos = np.array([c["pos"] for c in children], dtype=np.float64) * SCALE_ZYX

        # Vectorized pairwise physical distance matrix in micrometers
        dists = np.linalg.norm(p_pos[:, None, :] - c_pos[None, :, :], axis=-1)

        p_id_to_idx = {p["id"]: r for r, p in enumerate(parents)}
        c_id_to_idx = {c["id"]: col for col, c in enumerate(children)}

        probs = np.zeros_like(dists, dtype=np.float32)
        if edge_candidates:
            cands = edge_candidates.get(t_curr) or edge_candidates.get(str(t_curr))
            if not cands and str(t_curr).isdigit():
                cands = edge_candidates.get(int(t_curr))
            if cands:
                for c in cands:
                    sid, tid = c["src_id"], c["tgt_id"]
                    if sid in p_id_to_idx and tid in c_id_to_idx:
                        probs[p_id_to_idx[sid], c_id_to_idx[tid]] = c["prob"]

        if neural_weight > 0.0:
            cost_matrix = dists - neural_weight * (probs * 10.0)
            allowed = (dists <= max_dist_um) | ((dists <= max_prob_dist_um) & (probs >= prob_threshold))
        else:
            cost_matrix = dists
            allowed = dists <= max_dist_um

        cost_matrix = np.where(allowed, cost_matrix, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
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
    edge_candidates: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
    min_dist_um: float = MITOSIS_MIN_DIST_UM,
    max_dist_um: float = MITOSIS_MAX_DIST_UM,
    sister_min_um: float = MITOSIS_SISTER_MIN_UM,
    sister_max_um: float = MITOSIS_SISTER_MAX_UM,
    sym_diff_um: float = MITOSIS_SYM_DIFF_UM,
    cos_thresh: float = MITOSIS_COS_THRESH,
    min_mother_len: int = MITOSIS_MIN_MOTHER_LEN,
    min_prob: float = MITOSIS_MIN_PROB,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Biological Cytokinesis Mitosis Recovery with Forward Temporal Divergence.
    Recovers true 1 -> 2 daughter mitotic events while strictly rejecting false
    positives by enforcing:
      1. Mother interphase maturity: track history >= min_mother_len frames.
      2. Spindle symmetry: |d1 - d2| <= sym_diff_um.
      3. Cleavage collinearity: cos(theta) <= cos_thresh (>= 162 deg apart).
      4. Cleavage separation span: sister distance in [sister_min_um, sister_max_um].
      5. Neural edge support: max(prob(p, c1), prob(p, c2)) >= min_prob.
      6. Forward temporal divergence at t+2: daughters track into t+2 without contracting.
    """
    mitoses: List[Tuple[int, int, int]] = []
    final_edges = list(edges)
    edge_set = set(final_edges)

    def get_mother_track_len(node_id: int) -> int:
        curr = node_id
        length = 1
        while curr in bwd_links:
            curr = bwd_links[curr]
            length += 1
        return length

    frames = sorted(filtered_detections.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x)
    for i in range(len(frames) - 2):
        t_curr, t_next, t_grand = frames[i], frames[i + 1], frames[i + 2]
        try:
            if int(t_next) != int(t_curr) + 1 or int(t_grand) != int(t_curr) + 2:
                continue
        except (ValueError, TypeError):
            continue

        parents = filtered_detections[t_curr]
        children = filtered_detections[t_next]
        grand = filtered_detections[t_grand]
        if not parents or not children or not grand:
            continue

        c_by_id = {c["id"]: c for c in children}
        g_by_id = {g["id"]: g for g in grand}

        cand_probs = {}
        if edge_candidates:
            cands_t = (
                edge_candidates.get(t_curr)
                or edge_candidates.get(str(t_curr))
                or (edge_candidates.get(int(t_curr)) if str(t_curr).isdigit() else None)
                or []
            )
            cand_probs = {(c["src_id"], c["tgt_id"]): c["prob"] for c in cands_t}

        proposals = []
        for p in parents:
            pid = p["id"]
            if pid not in fwd_links:
                continue

            # 1. Mother cell interphase maturity constraint
            if get_mother_track_len(pid) < min_mother_len:
                continue

            c1_id = fwd_links[pid]
            c1 = c_by_id.get(c1_id)
            if not c1:
                continue

            p_pos = np.array(p["pos"])
            c1_pos = np.array(c1["pos"])
            v1 = (c1_pos - p_pos) * SCALE_ZYX
            d1 = float(np.linalg.norm(v1))
            if d1 < min_dist_um or d1 > max_dist_um:
                continue

            for c2 in children:
                c2_id = c2["id"]
                if c2_id == c1_id:
                    continue

                c2_pos = np.array(c2["pos"])
                v2 = (c2_pos - p_pos) * SCALE_ZYX
                d2 = float(np.linalg.norm(v2))
                if d2 < min_dist_um or d2 > max_dist_um:
                    continue

                sym_diff = abs(d1 - d2)
                if sym_diff > sym_diff_um:
                    continue

                cos_theta = float(np.dot(v1, v2) / (d1 * d2))
                if cos_theta > cos_thresh:
                    continue

                sister_d = float(np.linalg.norm((c1_pos - c2_pos) * SCALE_ZYX))
                if sister_d < sister_min_um or sister_d > sister_max_um:
                    continue

                # Neural edge support
                if cand_probs:
                    p1_prob = cand_probs.get((pid, c1_id), 0.0)
                    p2_prob = cand_probs.get((pid, c2_id), 0.0)
                    if max(p1_prob, p2_prob) < min_prob:
                        continue

                # Forward temporal divergence check at t+2
                g1_id = fwd_links.get(c1_id)
                g2_id = fwd_links.get(c2_id)
                if not g1_id or not g2_id or g1_id not in g_by_id or g2_id not in g_by_id:
                    continue

                g1_pos = np.array(g_by_id[g1_id]["pos"]) * SCALE_ZYX
                g2_pos = np.array(g_by_id[g2_id]["pos"]) * SCALE_ZYX
                g_dist = float(np.linalg.norm(g1_pos - g2_pos))
                if g_dist < sister_d - 0.5:
                    continue

                cost = sym_diff + 0.1 * abs(cos_theta - (-1.0))
                proposals.append((cost, pid, c1_id, c2_id))

        if proposals:
            proposals.sort(key=lambda x: x[0])
            cost, pid, d1_id, d2_id = proposals[0]
            # Disconnect intruder if any
            if d2_id in bwd_links:
                intruder = bwd_links[d2_id]
                if (intruder, d2_id) in edge_set:
                    final_edges.remove((intruder, d2_id))
                    edge_set.remove((intruder, d2_id))
            final_edges.append((pid, d2_id))
            edge_set.add((pid, d2_id))
            bwd_links[d2_id] = pid
            mitoses.append((pid, d1_id, d2_id))

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
    edge_candidates: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Main Graph Builder pipeline.
    Input:
      detections_by_frame: dict {t: list of dicts [{'id': int, 'pos': [z,y,x], 'conf': float, 'is_mitosis': float}]}
      estimated_nodes_per_frame: dict {t: int} from Zarr biological metadata.
      edge_candidates: dict {t: list of dicts [{'src_id': int, 'tgt_id': int, 'prob': float, 'dist_um': float}]}
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

    # Step 2: Frame-to-frame LAP tracking with neural gating
    edges, fwd_links, bwd_links = link_frame_to_frame(
        filtered_dets,
        max_dist_um=MAX_LINK_DISTANCE_UM,
        edge_candidates=edge_candidates,
        neural_weight=NEURAL_EDGE_WEIGHT,
        max_prob_dist_um=NEURAL_MAX_PROB_DIST_UM,
        prob_threshold=NEURAL_PROB_THRESHOLD,
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
            edge_candidates=edge_candidates,
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
