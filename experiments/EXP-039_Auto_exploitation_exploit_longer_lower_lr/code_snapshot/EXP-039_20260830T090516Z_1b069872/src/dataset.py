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
            return np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:]) # (Z, Y, X)
        except Exception:
            pass
    try:
        import zarr
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

class BiohubCellDataset(Dataset):
    def __init__(self, data_dir, sample_names, crop_size=(32, 128, 128), sigma=(1.0, 2.0, 2.0), is_train=True):
        self.data_dir = data_dir
        self.sample_names = sample_names
        self.crop_size = crop_size
        self.sigma = np.array(sigma, dtype=np.float32)
        self.is_train = is_train

        self.samples = []
        for name in sample_names:
            zarr_p = os.path.join(data_dir, f"{name}.zarr")
            geff_p = os.path.join(data_dir, f"{name}.geff")
            if os.path.exists(zarr_p) and os.path.exists(geff_p):
                with open(os.path.join(zarr_p, '0', 'zarr.json')) as f:
                    meta = json.load(f)
                shape = meta['shape'] # [T, Z, Y, X]
                dtype = np.dtype(meta['data_type'])
                nodes_by_t = load_geff_nodes(geff_p)
                for t in range(shape[0]):
                    if t in nodes_by_t and len(nodes_by_t[t]) > 0:
                        self.samples.append({
                            'name': name,
                            'zarr_path': zarr_p,
                            't': t,
                            'shape': shape,
                            'dtype': dtype,
                            'nodes': nodes_by_t[t]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        vol = load_zarr_chunk(item['zarr_path'], item['t'], item['shape'], item['dtype']).astype(np.float32)
        
        # Normalize volume
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
        vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

        Z, Y, X = vol.shape
        cz, cy, cx = self.crop_size

        if self.is_train:
            # Pick a center around an annotated node if possible
            if len(item['nodes']) > 0 and np.random.rand() > 0.2:
                node = item['nodes'][np.random.randint(len(item['nodes']))]
                z_start = int(np.clip(node[0] - cz // 2, 0, Z - cz))
                y_start = int(np.clip(node[1] - cy // 2, 0, Y - cy))
                x_start = int(np.clip(node[2] - cx // 2, 0, X - cx))
            else:
                z_start = np.random.randint(0, Z - cz + 1)
                y_start = np.random.randint(0, Y - cy + 1)
                x_start = np.random.randint(0, X - cx + 1)
        else:
            z_start, y_start, x_start = (Z - cz) // 2, (Y - cy) // 2, (X - cx) // 2

        vol_crop = vol[z_start:z_start+cz, y_start:y_start+cy, x_start:x_start+cx]

        # Generate Gaussian heatmap target
        heatmap = np.zeros((cz, cy, cx), dtype=np.float32)
        sz, sy, sx = self.sigma
        radius_z, radius_y, radius_x = int(3 * sz), int(3 * sy), int(3 * sx)

        for (nz, ny, nx) in item['nodes']:
            rz, ry, rx = nz - z_start, ny - y_start, nx - x_start
            if 0 <= rz < cz and 0 <= ry < cy and 0 <= rx < cx:
                z_min, z_max = max(0, rz - radius_z), min(cz, rz + radius_z + 1)
                y_min, y_max = max(0, ry - radius_y), min(cy, ry + radius_y + 1)
                x_min, x_max = max(0, rx - radius_x), min(cx, rx + radius_x + 1)

                zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
                dist_sq = ((zz - rz) / sz)**2 + ((yy - ry) / sy)**2 + ((xx - rx) / sx)**2
                gaussian = np.exp(-0.5 * dist_sq).astype(np.float32)
                heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                    heatmap[z_min:z_max, y_min:y_max, x_min:x_max], gaussian
                )

        if self.is_train:
            # 3D Data augmentations
            if np.random.rand() > 0.5:
                vol_crop = np.flip(vol_crop, axis=0)
                heatmap = np.flip(heatmap, axis=0)
            if np.random.rand() > 0.5:
                vol_crop = np.flip(vol_crop, axis=1)
                heatmap = np.flip(heatmap, axis=1)
            if np.random.rand() > 0.5:
                vol_crop = np.flip(vol_crop, axis=2)
                heatmap = np.flip(heatmap, axis=2)

        vol_tensor = torch.from_numpy(vol_crop.copy()).unsqueeze(0) # (1, Z, Y, X)
        heat_tensor = torch.from_numpy(heatmap.copy()).unsqueeze(0) # (1, Z, Y, X)

        return vol_tensor, heat_tensor
