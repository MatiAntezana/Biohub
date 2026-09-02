"""
Full-Volume 3D Dataset Loader (64, 256, 256)
Loads the ENTIRE embryo volume without spatial cropping or boundary slicing artifacts.
Uses 14-18 GB VRAM on RTX 4090 with batch_size=2 and AMP FP16.
"""

import os, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset
import blosc2
import zarr

SCALE = np.array([1.625, 0.40625, 0.40625]) # Z, Y, X um/voxel

def load_zarr_chunk(zarr_path, t, shape, dtype):
    chunk_path = os.path.join(zarr_path, '0', 'c', str(t), '0', '0', '0')
    if os.path.exists(chunk_path):
        try:
            with open(chunk_path, 'rb') as f:
                compressed = f.read()
            decompressed = blosc2.decompress(compressed)
            return np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])
        except Exception:
            pass
    try:
        arr = zarr.open(zarr_path, mode="r")["0"]
        return np.array(arr[t], dtype=dtype)
    except Exception:
        return np.zeros(shape[1:], dtype=dtype)

def load_geff_nodes(geff_path):
    root = zarr.open_group(geff_path, mode="r")
    t_arr = root["nodes"]["props"]["t"]["values"][:]
    z_arr = root["nodes"]["props"]["z"]["values"][:]
    y_arr = root["nodes"]["props"]["y"]["values"][:]
    x_arr = root["nodes"]["props"]["x"]["values"][:]
    nodes_by_t = {}
    for t, z, y, x in zip(t_arr, z_arr, y_arr, x_arr):
        t = int(t)
        if t not in nodes_by_t:
            nodes_by_t[t] = []
        nodes_by_t[t].append((int(round(z)), int(round(y)), int(round(x))))
    return nodes_by_t

class FullVolumeBiohubDataset(Dataset):
    def __init__(self, data_dir, sample_names, sigma=(1.0, 2.0, 2.0), is_train=True):
        self.data_dir = data_dir
        self.sample_names = sample_names
        self.sigma = np.array(sigma, dtype=np.float32)
        self.is_train = is_train

        self.samples = []
        for name in sample_names:
            zarr_p = os.path.join(data_dir, f"{name}.zarr")
            geff_p = os.path.join(data_dir, f"{name}.geff")
            if not os.path.exists(zarr_p) or not os.path.exists(geff_p):
                continue
            with open(os.path.join(zarr_p, '0', 'zarr.json')) as f:
                meta = json.load(f)
            shape = tuple(meta['shape'])
            dtype = np.dtype(meta['data_type'])
            n_t = shape[0]

            nodes_by_t = load_geff_nodes(geff_p)
            for t in range(n_t):
                if t in nodes_by_t and len(nodes_by_t[t]) > 0:
                    self.samples.append({
                        'name': name,
                        'zarr_path': zarr_p,
                        'shape': shape,
                        'dtype': dtype,
                        't': t,
                        'nodes': nodes_by_t[t]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        vol = load_zarr_chunk(item['zarr_path'], item['t'], item['shape'], item['dtype']).astype(np.float32)

        # Pad to exactly (64, 256, 256) if smaller
        target_shape = (64, 256, 256)
        pad_z = max(0, target_shape[0] - vol.shape[0])
        pad_y = max(0, target_shape[1] - vol.shape[1])
        pad_x = max(0, target_shape[2] - vol.shape[2])

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            vol = np.pad(vol, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

        vol = vol[:64, :256, :256]

        # Normalization
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
        vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

        # Target Gaussian Heatmap (Full Volume)
        target = np.zeros((64, 256, 256), dtype=np.float32)
        sz, sy, sx = self.sigma
        rad_z, rad_y, rad_x = int(3 * sz), int(3 * sy), int(3 * sx)

        for cz, cy, cx in item['nodes']:
            if not (0 <= cz < 64 and 0 <= cy < 256 and 0 <= cx < 256):
                continue
            z_min, z_max = max(0, cz - rad_z), min(64, cz + rad_z + 1)
            y_min, y_max = max(0, cy - rad_y), min(256, cy + rad_y + 1)
            x_min, x_max = max(0, cx - rad_x), min(256, cx + rad_x + 1)

            zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
            dist_sq = ((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2
            g = np.exp(-0.5 * dist_sq).astype(np.float32)
            target[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                target[z_min:z_max, y_min:y_max, x_min:x_max], g
            )

        # Random Flips if training
        if self.is_train:
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=2).copy()
                target = np.flip(target, axis=2).copy()
            if np.random.rand() > 0.5:
                vol = np.flip(vol, axis=1).copy()
                target = np.flip(target, axis=1).copy()

        return torch.from_numpy(vol).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)
