import os, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset
import blosc2
import zarr

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32) # Z, Y, X um/voxel

def load_zarr_chunk(zarr_path, t, shape, dtype):
    chunk_path = os.path.join(zarr_path, '0', 'c', str(t), '0', '0', '0')
    if not os.path.exists(chunk_path):
        chunk_path = os.path.join(zarr_path, '0', f"{t}.0.0.0")
    with open(chunk_path, 'rb') as f:
        compressed = f.read()
    decompressed = blosc2.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])

def load_geff_tracks_and_divisions(geff_path):
    root = zarr.open_group(geff_path, mode="r")
    t_arr = root["nodes"]["props"]["t"]["values"][:]
    z_arr = root["nodes"]["props"]["z"]["values"][:]
    y_arr = root["nodes"]["props"]["y"]["values"][:]
    x_arr = root["nodes"]["props"]["x"]["values"][:]
    ids_arr = root["nodes"]["ids"][:]

    node_dict = {}
    nodes_by_t = {}
    for nid, t, z, y, x in zip(ids_arr, t_arr, z_arr, y_arr, x_arr):
        t, nid = int(t), int(nid)
        pos = (int(round(z)), int(round(y)), int(round(x)))
        node_dict[nid] = (t, pos)
        if t not in nodes_by_t:
            nodes_by_t[t] = []
        nodes_by_t[t].append((nid, pos))

    # Read edges
    edge_sources = root["edges"]["ids"][0][:]
    edge_targets = root["edges"]["ids"][1][:]

    # Map forward edges: parent_nid -> list of daughter_nids
    fwd_edges = {}
    for s_id, t_id in zip(edge_sources, edge_targets):
        s_id, t_id = int(s_id), int(t_id)
        if s_id not in fwd_edges:
            fwd_edges[s_id] = []
        fwd_edges[s_id].append(t_id)

    return node_dict, nodes_by_t, fwd_edges

class BiohubMPMDataset(Dataset):
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
                shape = meta['shape']
                dtype = np.dtype(meta['data_type'])
                node_dict, nodes_by_t, fwd_edges = load_geff_tracks_and_divisions(geff_p)
                for t in range(shape[0] - 1):
                    if t in nodes_by_t and len(nodes_by_t[t]) > 0:
                        self.samples.append({
                            'name': name,
                            'zarr_path': zarr_p,
                            't': t,
                            'shape': shape,
                            'dtype': dtype,
                            'nodes': nodes_by_t[t],
                            'node_dict': node_dict,
                            'fwd_edges': fwd_edges
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        vol = load_zarr_chunk(item['zarr_path'], item['t'], item['shape'], item['dtype']).astype(np.float32)
        
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
        vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

        Z, Y, X = vol.shape
        cz, cy, cx = self.crop_size

        if self.is_train:
            if len(item['nodes']) > 0 and np.random.rand() > 0.2:
                _, node = item['nodes'][np.random.randint(len(item['nodes']))]
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

        # Targets
        heatmap = np.zeros((cz, cy, cx), dtype=np.float32)
        flow_target = np.zeros((3, cz, cy, cx), dtype=np.float32) # (dz, dy, dx) in physical microns
        flow_mask = np.zeros((cz, cy, cx), dtype=np.float32)
        div_target = np.zeros((cz, cy, cx), dtype=np.float32)

        sz, sy, sx = self.sigma
        radius_z, radius_y, radius_x = int(3 * sz), int(3 * sy), int(3 * sx)

        node_dict = item['node_dict']
        fwd_edges = item['fwd_edges']

        for (nid, (nz, ny, nx)) in item['nodes']:
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

                # Check forward motion
                if nid in fwd_edges and len(fwd_edges[nid]) > 0:
                    daughters = fwd_edges[nid]
                    # If 1 daughter: track motion
                    # If 2 daughters (division): motion to average centroid + mark division
                    if len(daughters) >= 2:
                        div_target[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                            div_target[z_min:z_max, y_min:y_max, x_min:x_max], gaussian
                        )
                    d_pos_list = [node_dict[d][1] for d in daughters if d in node_dict]
                    if d_pos_list:
                        avg_d_pos = np.mean(d_pos_list, axis=0) # (z, y, x)
                        phys_disp = (avg_d_pos - np.array([nz, ny, nx])) * SCALE # (dz, dy, dx) in um
                        for c_idx in range(3):
                            flow_target[c_idx, z_min:z_max, y_min:y_max, x_min:x_max] = phys_disp[c_idx]
                        flow_mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                            flow_mask[z_min:z_max, y_min:y_max, x_min:x_max], gaussian
                        )

        vol_t = torch.from_numpy(vol_crop.copy()).unsqueeze(0)
        heat_t = torch.from_numpy(heatmap.copy()).unsqueeze(0)
        flow_t = torch.from_numpy(flow_target.copy())
        fmask_t = torch.from_numpy(flow_mask.copy()).unsqueeze(0)
        div_t = torch.from_numpy(div_target.copy()).unsqueeze(0)

        return vol_t, heat_t, flow_t, fmask_t, div_t
