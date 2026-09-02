"""
Multi-Frame Gap-Closing & Trajectory Filtering Engine
Connects tracks across frame gaps (t -> t+2) and suppresses short orphan noise blips.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def track_with_gap_closing(
    nodes_by_t, # dict: {t: list of (nid, (z, y, x))}
    scale=np.array([1.625, 0.40625, 0.40625]),
    max_link_dist=14.0,
    enable_gap_closing=True,
    max_gap_dist=18.0,
    min_track_length=2,
    enable_mitosis=True,
    max_mitosis_dist=14.0,
    max_midpoint_dist=5.0
):
    """
    Returns:
      filtered_nodes: list of (nid, t, z, y, x)
      edges: list of (source_id, target_id)
    """
    all_timepoints = sorted(nodes_by_t.keys())
    if not all_timepoints:
        return [], []

    edges = [] # list of (pid, cid)
    fwd_links = {} # pid -> cid (or list of cids)
    bwd_links = {} # cid -> pid

    # 1. First Pass: Frame-by-Frame Physical Hungarian Matching (t -> t+1)
    for t_idx in range(len(all_timepoints) - 1):
        t1, t2 = all_timepoints[t_idx], all_timepoints[t_idx + 1]
        if t2 != t1 + 1:
            continue # Gap between frames handled in pass 2

        curr_nodes = nodes_by_t[t1]
        next_nodes = nodes_by_t[t2]

        if not curr_nodes or not next_nodes:
            continue

        c_ids = [n[0] for n in curr_nodes]
        n_ids = [n[0] for n in next_nodes]

        c_phys = np.array([n[1] for n in curr_nodes], dtype=np.float64) * scale
        n_phys = np.array([n[1] for n in next_nodes], dtype=np.float64) * scale

        dist_mat = np.sqrt(((c_phys[:, None] - n_phys[None, :]) ** 2).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(dist_mat)

        for r, c in zip(row_ind, col_ind):
            if dist_mat[r, c] <= max_link_dist:
                pid, cid = c_ids[r], n_ids[c]
                edges.append((pid, cid))
                fwd_links[pid] = [cid]
                bwd_links[cid] = pid

    # 2. Second Pass: Gap Closing (t -> t+2) for dead ends and new beginnings
    if enable_gap_closing:
        for t_idx in range(len(all_timepoints) - 2):
            t1, t3 = all_timepoints[t_idx], all_timepoints[t_idx + 2]
            if t3 != t1 + 2:
                continue

            curr_nodes = nodes_by_t[t1]
            future_nodes = nodes_by_t[t3]

            # Dead ends at t1 (nodes with no forward edge)
            dead_ends = [n for n in curr_nodes if n[0] not in fwd_links]
            # New beginnings at t3 (nodes with no backward edge)
            new_starts = [n for n in future_nodes if n[0] not in bwd_links]

            if dead_ends and new_starts:
                de_ids = [n[0] for n in dead_ends]
                ns_ids = [n[0] for n in new_starts]

                de_phys = np.array([n[1] for n in dead_ends], dtype=np.float64) * scale
                ns_phys = np.array([n[1] for n in new_starts], dtype=np.float64) * scale

                gap_dist_mat = np.sqrt(((de_phys[:, None] - ns_phys[None, :]) ** 2).sum(axis=2))
                row_ind, col_ind = linear_sum_assignment(gap_dist_mat)

                for r, c in zip(row_ind, col_ind):
                    if gap_dist_mat[r, c] <= max_gap_dist:
                        pid, cid = de_ids[r], ns_ids[c]
                        edges.append((pid, cid))
                        fwd_links[pid] = [cid]
                        bwd_links[cid] = pid

    # 3. Third Pass: Mitosis Detection with Temporal Persistence Gating
    if enable_mitosis:
        for t_idx in range(len(all_timepoints) - 2):
            t1, t2, t3 = all_timepoints[t_idx], all_timepoints[t_idx + 1], all_timepoints[t_idx + 2]
            if t2 != t1 + 1:
                continue

            curr_nodes = nodes_by_t[t1]
            next_nodes = nodes_by_t[t2]
            
            node_map_t1 = {n[0]: n[1] for n in curr_nodes}
            node_map_t2 = {n[0]: n[1] for n in next_nodes}

            # For each matched parent (p -> c1)
            for p_id in list(fwd_links.keys()):
                if p_id not in node_map_t1:
                    continue
                c1_id = fwd_links[p_id][0]
                if c1_id not in node_map_t2:
                    continue

                # Condition: c1 must persist to t3!
                if c1_id not in fwd_links:
                    continue

                p_pos = np.array(node_map_t1[p_id], dtype=np.float64) * scale
                d1_pos = np.array(node_map_t2[c1_id], dtype=np.float64) * scale

                # Search candidate daughter c2 among nodes at t2 that also persist to t3
                for n in next_nodes:
                    c2_id = n[0]
                    if c2_id == c1_id:
                        continue
                    # Persistence filter: c2 must survive to t3!
                    if c2_id not in fwd_links:
                        continue

                    d2_pos = np.array(n[1], dtype=np.float64) * scale
                    d_sep = np.linalg.norm(d1_pos - d2_pos)
                    
                    if not (2.5 <= d_sep <= 15.0):
                        continue
                    if np.linalg.norm(p_pos - d2_pos) > max_mitosis_dist:
                        continue

                    midpoint = (d1_pos + d2_pos) / 2.0
                    mid_err = np.linalg.norm(p_pos - midpoint)

                    v1 = d1_pos - p_pos
                    v2 = d2_pos - p_pos
                    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                    cos_a = np.dot(v1, v2) / max(1e-6, denom) if denom > 0 else 0.0

                    if mid_err <= max_midpoint_dist and cos_a < -0.3:
                        # Confirmed high-precision persistent mitosis!
                        edges.append((p_id, c2_id))
                        fwd_links[p_id].append(c2_id)
                        break

    # 4. Fourth Pass: Build Node List & Suppress Isolated Noise Blips
    all_active_nodes = set()
    for pid, cid in edges:
        all_active_nodes.add(pid)
        all_active_nodes.add(cid)

    filtered_nodes = []
    for t in all_timepoints:
        for nid, (z, y, x) in nodes_by_t[t]:
            # Keep nodes that participate in tracks or if min_track_length <= 1
            if min_track_length <= 1 or nid in all_active_nodes:
                filtered_nodes.append((nid, t, z, y, x))

    return filtered_nodes, edges
