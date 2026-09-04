"""
Mock validation cache generator.
Creates a biologically realistic 3D+t synthetic embryo dataset with cell migration,
mitotic divisions, optical anisotropy, and sparse ground truth annotations.
Allows instant, offline evaluation and benchmarking of Fast-Loop iterations.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def generate_synthetic_validation_dataset(
    n_timepoints: int = 40,
    n_initial_cells: int = 50,
    division_rate: float = 0.05,
    death_rate: float = 0.01,
    noise_sigma_voxels: float = 0.8,
    spurious_rate: float = 0.15,
    detection_miss_rate: float = 0.04,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generates realistic synthetic 3D cell tracks with divisions and optical anisotropy.
    Scale: Z=1.625 um/voxel, Y=X=0.40625 um/voxel.
    Returns a dictionary containing:
      - 'raw_detections': {t: list of dicts [{'id', 'pos', 'conf', 'is_mitosis'}]}
      - 'gt_nodes': {t: list of dicts [{'id', 'pos'}]}
      - 'gt_edges': list of tuples (source_id, target_id)
      - 'estimated_nodes_per_frame': {t: int}
      - 'metadata': dict with dataset properties
    """
    rng = np.random.RandomState(seed)
    
    # Volume bounds in voxels (Z: 64, Y: 256, X: 256)
    volume_bounds = np.array([64.0, 256.0, 256.0])
    
    # Initialize cells in the central region
    active_cells: List[Dict[str, Any]] = []
    next_node_id = 1
    
    for _ in range(n_initial_cells):
        pos = rng.uniform([10.0, 50.0, 50.0], [54.0, 206.0, 206.0])
        velocity = rng.normal(0.0, [0.3, 1.2, 1.2])  # Anisotropic velocity in voxels
        active_cells.append({
            "cell_lineage_id": next_node_id,
            "pos": pos,
            "velocity": velocity,
            "last_node_id": None,
            "age": rng.randint(0, 15),
        })
        next_node_id += 1

    gt_nodes_by_t: Dict[int, List[Dict[str, Any]]] = {}
    gt_edges: List[Tuple[int, int]] = []
    estimated_nodes_by_t: Dict[int, int] = {}
    raw_detections_by_t: Dict[int, List[Dict[str, Any]]] = {}
    
    det_id_counter = 100000

    for t in range(n_timepoints):
        current_gt_nodes = []
        new_active_cells = []
        
        # 1. Update existing cells and generate ground truth nodes
        for cell in active_cells:
            # Check cell death/disappearance
            if rng.uniform() < death_rate:
                continue
                
            # Update position
            cell["pos"] = np.clip(cell["pos"] + cell["velocity"], [2.0, 10.0, 10.0], volume_bounds - [2.0, 10.0, 10.0])
            cell["velocity"] = 0.85 * cell["velocity"] + 0.15 * rng.normal(0.0, [0.3, 1.0, 1.0])
            cell["age"] += 1
            
            node_id = next_node_id
            next_node_id += 1
            
            gt_node = {
                "id": node_id,
                "t": t,
                "pos": cell["pos"].copy(),
            }
            current_gt_nodes.append(gt_node)
            
            # Connect edge from previous timepoint
            if cell["last_node_id"] is not None:
                gt_edges.append((cell["last_node_id"], node_id))
            
            # Check for cell division (mitosis)
            if cell["age"] > 10 and rng.uniform() < division_rate:
                # Divide into two daughter cells
                # Separation vector along a random direction
                sep_dir = rng.normal(size=3)
                sep_dir[0] *= 0.25  # Anisotropic separation in Z
                sep_norm = np.linalg.norm(sep_dir)
                if sep_norm > 1e-4:
                    sep_dir /= sep_norm
                
                daughter_sep_dist = rng.uniform(3.0, 6.0)  # voxels
                d1_pos = np.clip(cell["pos"] + sep_dir * daughter_sep_dist, [2.0, 10.0, 10.0], volume_bounds - [2.0, 10.0, 10.0])
                d2_pos = np.clip(cell["pos"] - sep_dir * daughter_sep_dist, [2.0, 10.0, 10.0], volume_bounds - [2.0, 10.0, 10.0])
                
                d1 = {
                    "cell_lineage_id": next_node_id,
                    "pos": d1_pos,
                    "velocity": cell["velocity"] + rng.normal(0.0, [0.2, 0.8, 0.8]),
                    "last_node_id": node_id,  # Points to dividing mother
                    "age": 0,
                    "just_divided": True,
                }
                d2 = {
                    "cell_lineage_id": next_node_id + 1,
                    "pos": d2_pos,
                    "velocity": cell["velocity"] - rng.normal(0.0, [0.2, 0.8, 0.8]),
                    "last_node_id": node_id,  # Points to dividing mother
                    "age": 0,
                    "just_divided": True,
                }
                next_node_id += 2
                new_active_cells.extend([d1, d2])
            else:
                cell["last_node_id"] = node_id
                new_active_cells.append(cell)
                
        active_cells = new_active_cells
        gt_nodes_by_t[t] = current_gt_nodes
        
        # Ground truth total cells in this frame
        true_cell_count = len(current_gt_nodes)
        estimated_nodes_by_t[t] = true_cell_count
        
        # 2. Simulate raw detections (with positional noise, missed detections, and false peaks)
        raw_dets = []
        for g_node in current_gt_nodes:
            # Occasional missed detection
            if rng.uniform() < detection_miss_rate:
                continue
                
            noisy_pos = g_node["pos"] + rng.normal(0.0, noise_sigma_voxels, size=3)
            noisy_pos = np.clip(noisy_pos, [0.0, 0.0, 0.0], volume_bounds - 1.0)
            
            # Confidence score
            conf = float(np.clip(rng.beta(8, 2), 0.35, 0.99))
            
            # Is this cell about to divide or just divided?
            is_mitosis = 0.0
            # Check if this node in GT has 2 outgoing edges
            # We assign high is_mitosis probability to dividing nodes
            raw_dets.append({
                "id": det_id_counter,
                "t": t,
                "pos": noisy_pos,
                "conf": conf,
                "is_mitosis": is_mitosis,
                "gt_ref_id": g_node["id"],
            })
            det_id_counter += 1
            
        # 3. Add spurious false positive background peaks
        n_spurious = int(round(true_cell_count * spurious_rate))
        for _ in range(n_spurious):
            spurious_pos = rng.uniform([0.0, 0.0, 0.0], volume_bounds - 1.0)
            conf = float(np.clip(rng.beta(2, 5), 0.20, 0.65))
            raw_dets.append({
                "id": det_id_counter,
                "t": t,
                "pos": spurious_pos,
                "conf": conf,
                "is_mitosis": float(rng.uniform(0.0, 0.2)),
                "gt_ref_id": None,
            })
            det_id_counter += 1
            
        raw_detections_by_t[t] = raw_dets

    # Mark mitotic ground truth detections with elevated is_mitosis signal
    gt_outgoing_counts: Dict[int, int] = {}
    for u, v in gt_edges:
        gt_outgoing_counts[u] = gt_outgoing_counts.get(u, 0) + 1
    
    gt_division_ids = {nid for nid, cnt in gt_outgoing_counts.items() if cnt >= 2}
    
    for t, dets in raw_detections_by_t.items():
        for d in dets:
            if d.get("gt_ref_id") in gt_division_ids:
                d["is_mitosis"] = float(rng.uniform(0.70, 0.98))

    return {
        "raw_detections": raw_detections_by_t,
        "gt_nodes": gt_nodes_by_t,
        "gt_edges": gt_edges,
        "estimated_nodes_per_frame": estimated_nodes_by_t,
        "metadata": {
            "n_timepoints": n_timepoints,
            "total_gt_nodes": sum(len(nodes) for nodes in gt_nodes_by_t.values()),
            "total_gt_edges": len(gt_edges),
            "total_gt_divisions": len(gt_division_ids),
        }
    }


def save_dataset_to_cache(dataset: Dict[str, Any], output_path: Path | str) -> Path:
    """Serializes the dataset to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable = {
        "metadata": dataset["metadata"],
        "estimated_nodes_per_frame": {str(k): v for k, v in dataset["estimated_nodes_per_frame"].items()},
        "gt_edges": dataset["gt_edges"],
        "gt_nodes": {
            str(t): [
                {"id": int(n["id"]), "t": int(n["t"]), "pos": [float(x) for x in n["pos"]]}
                for n in nodes
            ]
            for t, nodes in dataset["gt_nodes"].items()
        },
        "raw_detections": {
            str(t): [
                {
                    "id": int(d["id"]),
                    "t": int(d["t"]),
                    "pos": [float(x) for x in d["pos"]],
                    "conf": float(d["conf"]),
                    "is_mitosis": float(d["is_mitosis"]),
                }
                for d in dets
            ]
            for t, dets in dataset["raw_detections"].items()
        },
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
        
    return path


def load_dataset_from_cache(cache_path: Path | str) -> Dict[str, Any]:
    """Loads dataset from cache and converts coordinates back to numpy arrays."""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_detections = {
        int(t): [
            {
                "id": int(d["id"]),
                "t": int(d["t"]),
                "pos": np.array(d["pos"], dtype=np.float64),
                "conf": float(d["conf"]),
                "is_mitosis": float(d["is_mitosis"]),
            }
            for d in dets
        ]
        for t, dets in data["raw_detections"].items()
    }
    
    gt_nodes = {
        int(t): [
            {
                "id": int(n["id"]),
                "t": int(n["t"]),
                "pos": np.array(n["pos"], dtype=np.float64),
            }
            for n in nodes
        ]
        for t, nodes in data["gt_nodes"].items()
    }
    
    gt_edges = [(int(u), int(v)) for u, v in data["gt_edges"]]
    estimated_nodes = {int(k): int(v) for k, v in data["estimated_nodes_per_frame"].items()}
    
    return {
        "raw_detections": raw_detections,
        "gt_nodes": gt_nodes,
        "gt_edges": gt_edges,
        "estimated_nodes_per_frame": estimated_nodes,
        "metadata": data.get("metadata", {}),
    }


if __name__ == "__main__":
    default_cache = Path(__file__).resolve().parent / "val_cache.json"
    print(f"Generating mock validation dataset at: {default_cache}")
    ds = generate_synthetic_validation_dataset()
    save_dataset_to_cache(ds, default_cache)
    print("Dataset generated successfully.")
    print(f"Metadata: {ds['metadata']}")
