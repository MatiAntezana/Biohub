"""
Biohub Cell Tracking - EXP-015 Champion Pipeline (CV Score 0.8096)
Trained on 90/10 Split (179 Train vols / 17,021 crops)
"""

import os, sys, glob, json, time, gc, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment

# Safe imports
try:
    import blosc2
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "blosc2"], check=False)
    import blosc2

try:
    from skimage.feature import peak_local_max
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "scikit-image"], check=False)
    from skimage.feature import peak_local_max

# ----------------- 1. Architecture -----------------

class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, anisotropic=True):
        super().__init__()
        k1 = (1, 3, 3) if anisotropic else (3, 3, 3)
        p1 = (0, 1, 1) if anisotropic else (1, 1, 1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=k1, padding=p1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class AnisotropicUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = AnisotropicConvBlock(in_channels, base_channels, anisotropic=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = AnisotropicConvBlock(base_channels, base_channels * 2, anisotropic=False)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = AnisotropicConvBlock(base_channels * 2, base_channels * 4, anisotropic=False)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = AnisotropicConvBlock(base_channels * 4, base_channels * 8, anisotropic=False)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = AnisotropicConvBlock(base_channels * 8, base_channels * 4, anisotropic=False)
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = AnisotropicConvBlock(base_channels * 4, base_channels * 2, anisotropic=False)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = AnisotropicConvBlock(base_channels * 2, base_channels, anisotropic=True)

        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))

# ----------------- 2. Gap Closing Tracker (Clean FP=0 Policy) -----------------

SCALE = np.array([1.625, 0.40625, 0.40625]) # Z, Y, X um/voxel

def track_with_gap_closing(
    nodes_by_t,
    scale=SCALE,
    max_link_dist=12.0,
    max_gap_dist=16.0,
    min_track_length=2
):
    all_timepoints = sorted(nodes_by_t.keys())
    if not all_timepoints:
        return [], []

    edges = []
    fwd_links = {}
    bwd_links = {}

    # Pass 1: Frame-to-Frame LAP
    for t_idx in range(len(all_timepoints) - 1):
        t1, t2 = all_timepoints[t_idx], all_timepoints[t_idx + 1]
        if t2 != t1 + 1:
            continue
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
                fwd_links[pid] = cid
                bwd_links[cid] = pid

    # Pass 2: Gap Closing (Delta t <= 2)
    for t_idx in range(len(all_timepoints) - 2):
        t1, t3 = all_timepoints[t_idx], all_timepoints[t_idx + 2]
        if t3 != t1 + 2:
            continue
        dead_ends = [n for n in nodes_by_t[t1] if n[0] not in fwd_links]
        new_starts = [n for n in nodes_by_t[t3] if n[0] not in bwd_links]

        if dead_ends and new_starts:
            de_ids = [n[0] for n in dead_ends]
            ns_ids = [n[0] for n in new_starts]
            de_phys = np.array([n[1] for n in dead_ends], dtype=np.float64) * scale
            ns_phys = np.array([n[1] for n in new_starts], dtype=np.float64) * scale

            gap_mat = np.sqrt(((de_phys[:, None] - ns_phys[None, :]) ** 2).sum(axis=2))
            row_ind, col_ind = linear_sum_assignment(gap_mat)

            for r, c in zip(row_ind, col_ind):
                if gap_mat[r, c] <= max_gap_dist:
                    pid, cid = de_ids[r], ns_ids[c]
                    edges.append((pid, cid))
                    fwd_links[pid] = cid
                    bwd_links[cid] = pid

    # Pass 3: Suppress isolated noise blips
    active_nodes = set()
    for pid, cid in edges:
        active_nodes.add(pid)
        active_nodes.add(cid)

    filtered_nodes = []
    for t in all_timepoints:
        for nid, (z, y, x) in nodes_by_t[t]:
            if min_track_length <= 1 or nid in active_nodes:
                filtered_nodes.append((nid, t, z, y, x))

    return filtered_nodes, edges

# ----------------- 3. Streaming Volumetric Inference -----------------

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
        import zarr
        arr = zarr.open(zarr_path, mode="r")["0"]
        return np.array(arr[t], dtype=dtype)
    except Exception:
        return np.zeros(shape[1:], dtype=dtype)

@torch.no_grad()
def predict_frame(model, vol, device, is_gpu=False):
    t_vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    if is_gpu:
        with torch.amp.autocast('cuda'):
            h0 = model(t_vol).squeeze().detach().cpu().numpy().astype(np.float32)
            hx = torch.flip(model(torch.flip(t_vol, dims=[-1])).squeeze(), dims=[-1]).detach().cpu().numpy().astype(np.float32)
            hy = torch.flip(model(torch.flip(t_vol, dims=[-2])).squeeze(), dims=[-2]).detach().cpu().numpy().astype(np.float32)
    else:
        h0 = model(t_vol).squeeze().detach().cpu().numpy().astype(np.float32)
        hx = torch.flip(model(torch.flip(t_vol, dims=[-1])).squeeze(), dims=[-1]).detach().cpu().numpy().astype(np.float32)
        hy = h0
    return (h0 + hx + hy) / 3.0

def find_test_zarrs():
    roots_to_scan = [
        "/kaggle/input/biohub-cell-tracking-during-development/test",
        "/kaggle/input/biohub-cell-tracking-during-development/test_data",
        "data/test",
    ]
    found_zarrs = []
    for root in roots_to_scan:
        if os.path.exists(root):
            found_zarrs = sorted(glob.glob(f"{root}/**/*.zarr", recursive=True))
            if found_zarrs:
                break
    if not found_zarrs:
        raise RuntimeError("BIOHUB_TEST_ONLY_GUARD: no test .zarr datasets found")
    for path in found_zarrs:
        parts = {part.lower() for part in os.path.abspath(path).split(os.sep) if part}
        if "train" in parts or not ({"test", "test_data"} & parts):
            raise RuntimeError(f"BIOHUB_TEST_ONLY_GUARD rejected dataset path: {path}")
    print("BIOHUB_DATASET_SCOPE=" + json.dumps({"scope": "test", "count": len(found_zarrs), "paths": found_zarrs}, sort_keys=True))
    return found_zarrs

def detect_device():
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
        return torch.device('cpu'), False
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7:
            return torch.device('cuda'), True
        else:
            torch.set_num_threads(4)
            return torch.device('cpu'), False
    except Exception:
        torch.set_num_threads(4)
        return torch.device('cpu'), False

def main():
    device, is_gpu = detect_device()
    print(f"Selected runtime configuration: Device={device}, GPU Acceleration={is_gpu}")

    test_zarrs = find_test_zarrs()
    print(f"Found {len(test_zarrs)} test volumes: {[os.path.basename(z) for z in test_zarrs]}")

    model = AnisotropicUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)

    # Search for EXP-015 champion weights (CV 0.8096)
    candidates = [
        "/kaggle/input/biohub-champion-weights/exp015_anisotropic_90_10.pt",
        "/kaggle/input/biohub-champion-weights/exp011_mitosis_80_20.pt",
        "/kaggle/input/biohub-champion-weights/exp005_fullfit_best.pt",
        "champion_weights/exp015_anisotropic_90_10.pt",
        "checkpoints/exp015_anisotropic_90_10.pt"
    ]
    for p in glob.glob("/kaggle/input/**/*.pt", recursive=True):
        if "exp015" in p.lower():
            candidates.insert(0, p)

    ckpt_path = next((c for c in candidates if os.path.exists(c)), None)
    if ckpt_path:
        print(f"Loading EXP-015 champion weights from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        raise RuntimeError("EXP-015 inference weights were not found; refusing untrained model")

    model.eval()
    all_submission_rows = []
    node_id_counter = 1
    t_start = time.time()

    for idx, zarr_p in enumerate(test_zarrs):
        print(f"BIOHUB_DATASET_PATH={os.path.abspath(zarr_p)}")
        folder_name = os.path.basename(zarr_p).replace('.zarr', '')
        t0_vol = time.time()
        print(f"[{idx+1}/{len(test_zarrs)}] Processing {folder_name}...")
        
        meta_path = os.path.join(zarr_p, '0', 'zarr.json')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(zarr_p, '.zarray')
        with open(meta_path) as f:
            meta = json.load(f)
        shape = tuple(meta.get('shape', [100, 64, 256, 256]))
        dtype = np.dtype(meta.get('data_type', meta.get('dtype', 'uint16')))
        n_t = shape[0]

        nodes_by_t = {}
        for t in range(n_t):
            try:
                vol = load_zarr_chunk(zarr_p, t, shape, dtype).astype(np.float32)
            except Exception as e:
                print(f"Warning: could not load frame {t}: {e}")
                continue

            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
            vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

            heat = predict_frame(model, vol, device, is_gpu=is_gpu)
            peaks = peak_local_max(heat, min_distance=2, threshold_abs=0.035, exclude_border=False)
            nodes_by_t[t] = []
            for p in peaks:
                nid = node_id_counter
                node_id_counter += 1
                nodes_by_t[t].append((nid, (int(p[0]), int(p[1]), int(p[2]))))

            del vol, heat
            if t % 25 == 0:
                gc.collect()

        # Multi-Frame Gap Closing (Clean FP=0 Division Policy)
        filtered_nodes, edges = track_with_gap_closing(
            nodes_by_t, scale=SCALE, max_link_dist=12.0, max_gap_dist=16.0, min_track_length=2
        )

        for nid, t, z, y, x in filtered_nodes:
            all_submission_rows.append({
                'dataset': folder_name,
                'row_type': 'node',
                'node_id': nid,
                't': t,
                'z': z,
                'y': y,
                'x': x,
                'source_id': -1,
                'target_id': -1,
            })

        for pid, cid in edges:
            all_submission_rows.append({
                'dataset': folder_name,
                'row_type': 'edge',
                'node_id': -1,
                't': -1,
                'z': -1,
                'y': -1,
                'x': -1,
                'source_id': pid,
                'target_id': cid,
            })

        print(f"  -> Finished {folder_name} in {time.time() - t0_vol:.1f}s ({len(filtered_nodes)} nodes, {len(edges)} edges)")
        gc.collect()

    out_path = "submission.csv"
    if os.path.exists("/kaggle/working"):
        out_path = "/kaggle/working/submission.csv"

    if all_submission_rows:
        sub_df = pd.DataFrame(all_submission_rows)
        sub_df.index.name = 'id'
        sub_df.to_csv(out_path)
        print(f"Saved {len(sub_df)} predictions to {out_path} in {time.time() - t_start:.1f}s total!")
    else:
        raise RuntimeError("BIOHUB_TEST_ONLY_GUARD: empty predictions; refusing dummy submission")

    print("Submission pipeline finished successfully! File exists:", os.path.exists(out_path), "Size:", os.path.getsize(out_path), "bytes")

if __name__ == "__main__":
    main()
