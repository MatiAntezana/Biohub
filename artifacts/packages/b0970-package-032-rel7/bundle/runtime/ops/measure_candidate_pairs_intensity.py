#!/usr/bin/env python3
"""Measure raw image intensity at midpoint of candidate gap pairs across development datasets."""
import json
from pathlib import Path
import numpy as np
import zarr

# The 9 candidate pairs <= 5.0 um identified in candidate graphs
CANDIDATE_PAIRS = [
    {"dataset": "6bba_09961292", "u": 11739, "v": 12235, "t": 38, "dist_um": 1.72357, "pos_u": [1., 120., 152.], "pos_v": [1., 123., 155.]},
    {"dataset": "44b6_c50204e0", "u": 31510, "v": 32172, "t": 91, "dist_um": 2.03125, "pos_u": [37., 141., 79.], "pos_v": [37., 146., 79.]},
    {"dataset": "6bba_09961292", "u": 11712, "v": 12210, "t": 38, "dist_um": 2.87262, "pos_u": [0., 52., 196.], "pos_v": [0., 57., 191.]},
    {"dataset": "44b6_c50204e0", "u": 24239, "v": 24954, "t": 71, "dist_um": 3.56483, "pos_u": [32., 210., 185.], "pos_v": [34., 209., 177.]},
    {"dataset": "44b6_ddf577ad", "u": 29440, "v": 30182, "t": 82, "dist_um": 3.56483, "pos_u": [45., 160., 187.], "pos_v": [45., 153., 192.]},
    {"dataset": "44b6_e57ff5c6", "u": 5215, "v": 5563, "t": 33, "dist_um": 3.65625, "pos_u": [29., 139., 225.], "pos_v": [29., 148., 224.]},
    {"dataset": "44b6_d5e7d891", "u": 19524, "v": 20365, "t": 52, "dist_um": 3.83255, "pos_u": [31., 135., 230.], "pos_v": [31., 144., 233.]},
    {"dataset": "6bba_48816121", "u": 13337, "v": 13828, "t": 47, "dist_um": 4.61411, "pos_u": [31., 201., 209.], "pos_v": [30., 204., 198.]},
    {"dataset": "44b6_c50204e0", "u": 4317, "v": 4898, "t": 16, "dist_um": 4.90874, "pos_u": [40., 204., 163.], "pos_v": [42., 210., 153.]},
]

DATA_DIR = Path("/home/alumno1/Documents/Biohub/data/train")

results = []
for p in CANDIDATE_PAIRS:
    ds = p["dataset"]
    z_path = DATA_DIR / f"{ds}.zarr"
    if not z_path.exists():
        continue
    z_arr = zarr.open(z_path, mode="r")["0"]
    
    t_mid = p["t"] + 1
    pos_u = np.array(p["pos_u"])
    pos_v = np.array(p["pos_v"])
    r_mid = 0.5 * (pos_u + pos_v)
    iz, iy, ix = int(round(r_mid[0])), int(round(r_mid[1])), int(round(r_mid[2]))
    
    z_min, z_max = max(0, iz - 3), min(z_arr.shape[1], iz + 4)
    y_min, y_max = max(0, iy - 7), min(z_arr.shape[2], iy + 8)
    x_min, x_max = max(0, ix - 7), min(z_arr.shape[3], ix + 8)
    
    patch = np.array(z_arr[t_mid, z_min:z_max, y_min:y_max, x_min:x_max], dtype=float)
    patch_mean = float(patch.mean()) if patch.size > 0 else 0.0
    patch_std = float(patch.std()) if patch.size > 0 else 1.0
    patch_max = float(patch.max()) if patch.size > 0 else 0.0
    
    val_at_mid = float(z_arr[t_mid, min(max(0, iz), z_arr.shape[1]-1),
                             min(max(0, iy), z_arr.shape[2]-1),
                             min(max(0, ix), z_arr.shape[3]-1)])
    
    norm_val = (val_at_mid - patch_mean) / max(patch_std, 1e-4)
    snr_max = (patch_max - patch_mean) / max(patch_std, 1e-4)
    ratio_to_max = val_at_mid / max(patch_max, 1e-4)
    
    res = {
        "dataset": ds,
        "u": p["u"],
        "v": p["v"],
        "t_mid": t_mid,
        "dist_um": p["dist_um"],
        "r_mid_vox": [float(iz), float(iy), float(ix)],
        "val_at_mid": val_at_mid,
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "patch_max": patch_max,
        "norm_intensity_zscore": norm_val,
        "snr_patch_max": snr_max,
        "ratio_to_max": ratio_to_max,
        "confirms_cell": bool(norm_val > 2.0 and ratio_to_max > 0.85),
    }
    results.append(res)
    print(json.dumps(res))

out_path = Path("/home/alumno1/Documents/Biohub-0970/runs/candidate_pairs_intensity.json")
out_path.write_text(json.dumps(results, indent=2))
print("Saved to", out_path)
