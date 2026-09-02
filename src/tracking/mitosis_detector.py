import numpy as np
from scipy.optimize import linear_sum_assignment

def link_and_detect_mitosis(
    prev_centroids, # dict {nid: (z, y, x)}
    curr_centroids, # dict {nid: (z, y, x)}
    scale=np.array([1.625, 0.40625, 0.40625]),
    max_link_dist=12.0,
    enable_mitosis=True,
    max_mitosis_dist=14.0,
    max_midpoint_dist=5.0,
    min_daughter_dist=2.5,
    max_daughter_dist=15.0
):
    """
    Two-stage LAP tracking with geometric mitosis (1 -> 2) branching.
    Stage 1: Hungarian 1-to-1 linear sum assignment.
    Stage 2: For matched parents, evaluate if an unmatched curr node is a second daughter.
    """
    edges = []
    if not prev_centroids or not curr_centroids:
        return edges

    prev_ids = list(prev_centroids.keys())
    curr_ids = list(curr_centroids.keys())

    prev_coords = np.array([prev_centroids[pid] for pid in prev_ids], dtype=np.float64)
    curr_coords = np.array([curr_centroids[cid] for cid in curr_ids], dtype=np.float64)

    prev_phys = prev_coords * scale
    curr_phys = curr_coords * scale

    # Physical Euclidean distance matrix
    dist_mat = np.sqrt(((prev_phys[:, None] - curr_phys[None, :]) ** 2).sum(axis=2))

    # Stage 1: Standard 1-to-1 matching
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    
    matched_prev = set()
    matched_curr = set()
    primary_matches = {} # pid -> cid

    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] <= max_link_dist:
            pid = prev_ids[r]
            cid = curr_ids[c]
            matched_prev.add(r)
            matched_curr.add(c)
            primary_matches[r] = c
            edges.append((pid, cid))

    if not enable_mitosis:
        return edges

    # Stage 2: Evaluate unmatched curr nodes as second daughters of matched parents
    unmatched_curr = [c for c in range(len(curr_ids)) if c not in matched_curr]
    
    for r, c1 in list(primary_matches.items()):
        pid = prev_ids[r]
        p_pos = prev_phys[r]
        d1_pos = curr_phys[c1]

        best_c2 = None
        best_mid_err = float('inf')

        for c2 in unmatched_curr:
            d2_pos = curr_phys[c2]
            d_d1_d2 = np.linalg.norm(d1_pos - d2_pos)
            
            # Condition: daughters must be separated by reasonable distance
            if not (min_daughter_dist <= d_d1_d2 <= max_daughter_dist):
                continue
            
            # Condition: daughter 2 must be close to parent
            if np.linalg.norm(p_pos - d2_pos) > max_mitosis_dist:
                continue

            # Condition: midpoint of d1 and d2 must be close to parent
            midpoint = (d1_pos + d2_pos) / 2.0
            mid_err = np.linalg.norm(p_pos - midpoint)
            
            # Condition: angle between (d1 - p) and (d2 - p) should be diverging (> 90 deg, cos < 0)
            v1 = d1_pos - p_pos
            v2 = d2_pos - p_pos
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            cos_angle = np.dot(v1, v2) / max(1e-6, denom) if denom > 0 else 0.0

            if mid_err <= max_midpoint_dist and cos_angle < -0.2 and mid_err < best_mid_err:
                best_mid_err = mid_err
                best_c2 = c2

        if best_c2 is not None:
            # Second daughter confirmed!
            cid2 = curr_ids[best_c2]
            edges.append((pid, cid2))
            unmatched_curr.remove(best_c2)

    return edges
