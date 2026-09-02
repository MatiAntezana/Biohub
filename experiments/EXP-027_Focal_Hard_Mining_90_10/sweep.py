import os, sys, glob, json, time, subprocess
import numpy as np, pandas as pd
import torch, torch.nn as nn
from skimage.feature import peak_local_max

from src.dataset import load_zarr_chunk, SCALE
from src.models.unet3d_anisotropic import AnisotropicUNet3D
from src.tracking.gap_closing_tracker import track_with_gap_closing

def run_eval(peak_thresh=0.035):
    device = torch.device('cuda')
    model = AnisotropicUNet3D(in_channels=1, out_channels=1, base_channels=64).to(device)
    ckpt = torch.load('/home/alumno1/Documents/Biohub/checkpoints/exp027_focal_alpha20_90_10.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    val_samples = ['44b6_0b24845f', '44b6_c8e2a523', '6bba_07477033', '6bba_20852818', '6bba_3a1849c2', '6bba_3fda6b25', '6bba_5c824876', '6bba_74686d6a', '6bba_971fa5e0', '6bba_9e23430b']
    train_dir = 'data/train'
    
    all_submission_rows = []
    node_id_counter = 1

    for s in val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        with open(os.path.join(z_path, '0', 'zarr.json')) as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['data_type'])
        n_t = shape[0]

        nodes_by_t = {}
        for t in range(n_t):
            vol = load_zarr_chunk(z_path, t, shape, dtype).astype(np.float32)
            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
            vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

            vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                h0 = model(vol_t).squeeze().detach().cpu().numpy().astype(np.float32)
                hx = torch.flip(model(torch.flip(vol_t, dims=[-1])).squeeze(), dims=[-1]).detach().cpu().numpy().astype(np.float32)
            heat = (h0 + hx) / 2.0

            peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
            nodes_by_t[t] = []
            for p in peaks:
                nid = node_id_counter
                node_id_counter += 1
                nodes_by_t[t].append((nid, (int(p[0]), int(p[1]), int(p[2]))))

        filtered_nodes, edges = track_with_gap_closing(
            nodes_by_t, scale=SCALE, max_link_dist=12.0, max_gap_dist=16.0, min_track_length=2, enable_mitosis=False
        )

        for nid, t, z, y, x in filtered_nodes:
            all_submission_rows.append({'dataset': s, 'row_type': 'node', 'node_id': nid, 't': t, 'z': z, 'y': y, 'x': x, 'source_id': -1, 'target_id': -1})

        for pid, cid in edges:
            all_submission_rows.append({'dataset': s, 'row_type': 'edge', 'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1, 'source_id': pid, 'target_id': cid})

    sub_df = pd.DataFrame(all_submission_rows)
    sub_df.index.name = 'id'
    sub_path = f'sweep_sub_exp027_{peak_thresh}.csv'
    sub_df.to_csv(sub_path)

    cmd = ['python3', '-m', 'src.metrics.evaluate', sub_path, 'data/train']
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if 'score=' in line:
            print(f"Threshold tau={peak_thresh:.3f} -> {line.strip()}")

if __name__ == '__main__':
    for t in [0.010, 0.020, 0.035, 0.050, 0.070, 0.100]:
        run_eval(t)
