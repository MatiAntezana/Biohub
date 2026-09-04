import json
import numpy as np
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

with open(PROJECT_ROOT / "fast_loop" / "val_cache.json") as f:
    data = json.load(f)

gt_edges = data["gt_edges"]
gt_nodes = {}
for t_str, cell_list in data["gt_nodes"].items():
    for n in cell_list:
        gt_nodes[n["id"]] = n

print(f"Total GT nodes: {len(gt_nodes)}, Total GT edges: {len(gt_edges)}")

parent_counts = {}
for u, v in gt_edges:
    parent_counts.setdefault(u, []).append(v)

divisions = [(p, ch) for p, ch in parent_counts.items() if len(ch) >= 2]
print(f"GT divisions found: {len(divisions)}")
scale = np.array([1.625, 0.40625, 0.40625])

for p, ch in divisions:
    p_node = gt_nodes[p]
    p_pos = np.array(p_node["pos"])
    print(f"Parent {p} at t={p_node['t']}, pos={p_pos}")
    for c in ch:
        c_node = gt_nodes[c]
        c_pos = np.array(c_node["pos"])
        d = np.linalg.norm((p_pos - c_pos) * scale)
        print(f"  -> Child {c} at t={c_node['t']}, pos={c_pos}, dist_um={d:.3f}")
    c1_pos = np.array(gt_nodes[ch[0]]["pos"])
    c2_pos = np.array(gt_nodes[ch[1]]["pos"])
    sister_d = np.linalg.norm((c1_pos - c2_pos) * scale)
    v1 = (c1_pos - p_pos) * scale
    v2 = (c2_pos - p_pos) * scale
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"  Sister distance: {sister_d:.3f} um")
    print(f"  v1.v2 cos(angle): {cos_angle:.3f} (Angle: {np.degrees(np.arccos(np.clip(cos_angle, -1, 1))):.1f} deg)")

# Now test this geometric filter across ALL raw detections in the dataset
raw_dets = {int(k): v for k, v in data["raw_detections"].items()}
est_nodes = {int(k): v for k, v in data["estimated_nodes_per_frame"].items()}
# Re-run LAP tracking as in baseline
from core.graph_builder import apply_node_budgeting, link_frame_to_frame

filtered_dets = apply_node_budgeting(raw_dets, est_nodes)
edges, fwd_links, bwd_links = link_frame_to_frame(filtered_dets, max_dist_um=5.5)

print("\n--- Testing Cytokinesis Opposing Direction Filter on Raw Detections ---")
candidate_divisions = []
frames = sorted([int(k) for k in filtered_dets.keys()])
for i in range(len(frames) - 1):
    t1, t2 = frames[i], frames[i+1]
    parents = filtered_dets[t1]
    children = filtered_dets[t2]
    unmatched = [c for c in children if c["id"] not in bwd_links]
    
    for p in parents:
        pid = p["id"]
        if pid not in fwd_links:
            continue
        c1_id = fwd_links[pid]
        c1 = next((c for c in children if c["id"] == c1_id), None)
        if not c1:
            continue
        p_pos = np.array(p["pos"])
        c1_pos = np.array(c1["pos"])
        v1 = (c1_pos - p_pos) * scale
        norm_v1 = np.linalg.norm(v1)
        if norm_v1 < 2.0 or norm_v1 > 5.5:
            continue
            
        for c2 in unmatched:
            c2_pos = np.array(c2["pos"])
            v2 = (c2_pos - p_pos) * scale
            norm_v2 = np.linalg.norm(v2)
            if norm_v2 < 2.0 or norm_v2 > 5.5:
                continue
            # Symmetry
            if abs(norm_v1 - norm_v2) > 0.8:
                continue
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            if cos_theta > -0.7:  # Must be nearly collinear in opposite directions!
                continue
            # Sister distance
            sister_d = np.linalg.norm((c1_pos - c2_pos) * scale)
            if sister_d < 6.0 or sister_d > 11.0:
                continue
            candidate_divisions.append({
                "t": t1,
                "pid": pid,
                "c1": c1_id,
                "c2": c2["id"],
                "cos_theta": float(cos_theta),
                "diff": float(abs(norm_v1 - norm_v2)),
                "d1": float(norm_v1),
                "d2": float(norm_v2),
                "sister_d": float(sister_d),
            })

# Find detected nodes corresponding to the true division
gt_p = gt_nodes[172000000050]
gt_c1 = gt_nodes[173000000050]
gt_c2 = gt_nodes[173000000051]

def find_nearest_detection(pos, dets):
    best_d = 1e9
    best_c = None
    for c in dets:
        d = np.linalg.norm((np.array(pos) - np.array(c["pos"])) * scale)
        if d < best_d:
            best_d = d
            best_c = c
    return best_c, best_d

det_p, d_p = find_nearest_detection(gt_p["pos"], filtered_dets[66])
det_c1, d_c1 = find_nearest_detection(gt_c1["pos"], filtered_dets[67])
det_c2, d_c2 = find_nearest_detection(gt_c2["pos"], filtered_dets[67])

print(f"\nGT Parent matched to det {det_p['id']} (dist {d_p:.3f}um, conf {det_p['conf']:.2f})")
print(f"GT Child 1 matched to det {det_c1['id']} (dist {d_c1:.3f}um, conf {det_c1['conf']:.2f})")
print(f"GT Child 2 matched to det {det_c2['id']} (dist {d_c2:.3f}um, conf {det_c2['conf']:.2f})")

print(f"det_p in fwd_links? {det_p['id'] in fwd_links} -> fwd_link={fwd_links.get(det_p['id'])}")
print(f"det_c1 in bwd_links? {det_c1['id'] in bwd_links} -> bwd_link={bwd_links.get(det_c1['id'])}")
print(f"det_c2 in bwd_links? {det_c2['id'] in bwd_links} -> bwd_link={bwd_links.get(det_c2['id'])}")

print("\n--- Testing Motion Momentum Tracking ---")
from scipy.optimize import linear_sum_assignment
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import load_dataset_from_cache

ds = load_dataset_from_cache(PROJECT_ROOT / "fast_loop" / "val_cache.json")
filtered_dets = apply_node_budgeting(ds["raw_detections"], ds["estimated_nodes_per_frame"])
total_est = sum(ds["estimated_nodes_per_frame"].values())

def link_with_momentum(filtered_dets, max_dist_um=5.5, beta=0.3):
    edges = []
    fwd_links = {}
    bwd_links = {}
    prev_pos_by_id = {}
    
    frames = sorted(filtered_dets.keys())
    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i + 1]
        if t_next != t_curr + 1:
            continue
        parents = filtered_dets[t_curr]
        children = filtered_dets[t_next]
        
        pred_p_pos = []
        for p in parents:
            pid = p["id"]
            pos = np.array(p["pos"], dtype=np.float64)
            if pid in prev_pos_by_id:
                v = pos - prev_pos_by_id[pid]
                v_um = np.linalg.norm(v * scale)
                if v_um > 3.0:
                    v = v * (3.0 / v_um)
                pred_pos = pos + beta * v
            else:
                pred_pos = pos
            pred_p_pos.append(pred_pos)
            
        pred_p_pos = np.array(pred_p_pos, dtype=np.float64) * scale
        c_pos = np.array([c["pos"] for c in children], dtype=np.float64) * scale
        
        dists = np.linalg.norm(pred_p_pos[:, None, :] - c_pos[None, :, :], axis=-1)
        cost_matrix = np.where(dists <= max_dist_um, dists, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_dist_um:
                pid = parents[r]["id"]
                cid = children[c]["id"]
                edges.append((pid, cid))
                fwd_links[pid] = cid
                bwd_links[cid] = pid
                prev_pos_by_id[cid] = np.array(parents[r]["pos"], dtype=np.float64)
                
    return edges, fwd_links, bwd_links

for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    ed, fwd, bwd = link_with_momentum(filtered_dets, max_dist_um=5.5, beta=beta)
    tel = evaluate_tracking(
        pred_nodes_by_t=filtered_dets,
        pred_edges=ed,
        gt_nodes_by_t=ds["gt_nodes"],
        gt_edges=ds["gt_edges"],
        estimated_total_nodes=total_est,
    )
    print(f"beta={beta:.2f} | Score={tel.total_score:.4f} | AdjEdge={tel.adjusted_edge_jaccard:.4f} | TP={tel.edge_tp:3d} FP={tel.edge_fp:3d} FN={tel.edge_fn:3d} | Frag={tel.edges_fragmented:2d}")





