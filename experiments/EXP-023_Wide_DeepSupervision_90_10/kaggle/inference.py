"""
BIOHUB CELL TRACKING - EXP-023 3D WIDE DEEP SUPERVISION (64ch) SUBMISSION ENGINE
Architecture: 3D Wide Deep Supervision U-Net 64ch + Fast Gated LAP Tracking (CV 0.8779)
Target: Tesla T4 execution in < 3 minutes commit, < 18 minutes hidden test.
"""

import os, sys, glob, json, time, gc, shutil, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Launching Biohub EXP-023 Wide Deep Supervision (64ch) Submission ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()
print(f"Device: {device} ({torch.cuda.get_device_name(0) if is_gpu else 'CPU'})")

# ----------------- 0. Install Offline Packages if available -----------------

wheel_dirs = glob.glob("/kaggle/input/**/wheels", recursive=True) + glob.glob("champion_weights/wheels")
for wd in wheel_dirs:
    print(f"Found offline wheels in: {wd}")
    os.system(f"pip install --no-index --find-links={wd} zarr blosc2 numcodecs >/dev/null 2>&1")

# ----------------- 1. Architecture Definitions -----------------

class DeepSupervisionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.head_aux3 = nn.Conv3d(base_channels * 4, out_channels, kernel_size=1)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.head_aux2 = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.head_main = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        b = self.bottleneck(p3)

        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))

        main_out = torch.sigmoid(self.head_main(d1))
        return main_out

# ----------------- 2. Direct Safe Frame Loader -----------------

def get_frame(arr, zarr_path, t, shape=(100, 64, 256, 256), dtype=np.dtype('uint16')):
    if arr is not None:
        try:
            vol = np.array(arr[t], dtype=np.float32)
            if vol.shape == (64, 256, 256):
                p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
                return np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)
        except Exception:
            pass

    for sub in [f"0/c/{t}/0/0/0", f"0/c/{t}", f"0/{t}.0.0.0", f"0/{t}/0/0/0", f"0/{t}"]:
        p = os.path.join(zarr_path, sub)
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    raw = f.read()
                try:
                    import blosc2
                    decomp = blosc2.decompress(raw)
                    vol = np.frombuffer(decomp, dtype=dtype).reshape((64, 256, 256)).astype(np.float32)
                except Exception:
                    vol = np.frombuffer(raw, dtype=dtype).reshape((64, 256, 256)).astype(np.float32)
                p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
                return np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)
            except Exception:
                pass

    return np.zeros((64, 256, 256), dtype=np.float32)

# ----------------- 3. Model Weight Loader -----------------

def find_weights_file():
    candidates = [
        "/kaggle/input/datasets/matiasantezana/biohub-champion-weights/exp023_wide_deep_supervision_90_10.pt",
        "/kaggle/input/biohub-champion-weights/exp023_wide_deep_supervision_90_10.pt",
        "./champion_weights/exp023_wide_deep_supervision_90_10.pt",
        "../champion_weights/exp023_wide_deep_supervision_90_10.pt"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    found = glob.glob("/kaggle/input/**/exp023*.pt", recursive=True) or glob.glob("./**/exp023*.pt", recursive=True)
    return found[0] if found else None

def load_exp023_model(weights_path, device):
    print(f"Loading EXP-023 model weights from: {weights_path}")
    model = DeepSupervisionUNet3D(in_channels=1, out_channels=1, base_channels=64)
    state = torch.load(weights_path, map_location="cpu")
    if "model_state" in state:
        model.load_state_dict(state["model_state"], strict=False)
    elif "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("Successfully loaded EXP-023 Wide Deep Supervision model.")
    return model

# ----------------- 4. Calibrated Fast Peak Extraction & Gated Tracking -----------------

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64) # Z, Y, X physical scales in um

@torch.no_grad()
def extract_peaks_3d(heat_tensor, thresh=0.050, min_dist=3, max_peaks=500):
    pad = min_dist
    kernel_size = 2 * min_dist + 1
    max_pooled = F.max_pool3d(heat_tensor, kernel_size=kernel_size, stride=1, padding=pad)
    is_peak = (heat_tensor == max_pooled) & (heat_tensor >= thresh)
    coords = torch.nonzero(is_peak.squeeze(0).squeeze(0), as_tuple=False)
    coords_np = coords.cpu().numpy()
    if len(coords_np) > max_peaks:
        values = heat_tensor.squeeze(0).squeeze(0)[coords[:, 0], coords[:, 1], coords[:, 2]].cpu().numpy()
        top_idx = np.argsort(values)[-max_peaks:]
        coords_np = coords_np[top_idx]
    return coords_np

def track_with_gap_closing(nodes_by_t, scale=SCALE, max_link_dist=12.0, max_gap_dist=16.0, min_track_length=2):
    all_timepoints = sorted(nodes_by_t.keys())
    if not all_timepoints:
        return [], []

    edges = []
    fwd_links = {}
    bwd_links = {}

    # Pass 1: Frame-to-Frame LAP (t -> t+1)
    for t_idx in range(len(all_timepoints) - 1):
        t1, t2 = all_timepoints[t_idx], all_timepoints[t_idx + 1]
        if t2 != t1 + 1:
            continue
        curr_nodes = nodes_by_t[t1]
        next_nodes = nodes_by_t[t2]
        if not curr_nodes or not next_nodes:
            continue

        c_ids = [n["id"] for n in curr_nodes]
        n_ids = [n["id"] for n in next_nodes]
        c_phys = np.array([n["pos"] for n in curr_nodes], dtype=np.float64) * scale
        n_phys = np.array([n["pos"] for n in next_nodes], dtype=np.float64) * scale

        dist_mat = np.sqrt(((c_phys[:, None] - n_phys[None, :]) ** 2).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(dist_mat)

        for r, c in zip(row_ind, col_ind):
            if dist_mat[r, c] <= max_link_dist:
                pid, cid = c_ids[r], n_ids[c]
                edges.append((pid, cid))
                fwd_links[pid] = cid
                bwd_links[cid] = pid

    # Pass 2: Gap Closing (t -> t+2)
    for t_idx in range(len(all_timepoints) - 2):
        t1, t3 = all_timepoints[t_idx], all_timepoints[t_idx + 2]
        if t3 != t1 + 2:
            continue
        dead_ends = [n for n in nodes_by_t[t1] if n["id"] not in fwd_links]
        new_starts = [n for n in nodes_by_t[t3] if n["id"] not in bwd_links]

        if dead_ends and new_starts:
            de_ids = [n["id"] for n in dead_ends]
            ns_ids = [n["id"] for n in new_starts]
            de_phys = np.array([n["pos"] for n in dead_ends], dtype=np.float64) * scale
            ns_phys = np.array([n["pos"] for n in new_starts], dtype=np.float64) * scale

            gap_mat = np.sqrt(((de_phys[:, None] - ns_phys[None, :]) ** 2).sum(axis=2))
            row_ind, col_ind = linear_sum_assignment(gap_mat)

            for r, c in zip(row_ind, col_ind):
                if gap_mat[r, c] <= max_gap_dist:
                    pid, cid = de_ids[r], ns_ids[c]
                    edges.append((pid, cid))
                    fwd_links[pid] = cid
                    bwd_links[cid] = pid

    all_parent_ids = set(p for p, c in edges)
    all_child_ids = set(c for p, c in edges)
    connected_ids = all_parent_ids.union(all_child_ids)

    final_nodes = []
    for t in all_timepoints:
        for n in nodes_by_t[t]:
            if min_track_length <= 1 or n["id"] in connected_ids:
                final_nodes.append(n)

    return final_nodes, edges

# ----------------- 5. Test Dataset Processing Loop -----------------

def find_test_data_dir():
    candidates = [
        "/kaggle/input/biohub-cell-tracking-during-development/test",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
        "./data/test",
        "../data/test"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

weights_path = find_weights_file()
test_data_dir = find_test_data_dir()

print(f"Weights path: {weights_path}")
print(f"Test data dir: {test_data_dir}")

model = load_exp023_model(weights_path, device) if weights_path else None

test_zarrs = sorted(glob.glob(f"{test_data_dir}/*.zarr")) if test_data_dir else []
print(f"Total test datasets to process: {len(test_zarrs)}")

all_rows = []
total_nodes_found = 0
total_edges_found = 0
start_time = time.time()

for zarr_idx, zarr_path in enumerate(test_zarrs):
    ds_name = os.path.basename(zarr_path).replace(".zarr", "")
    t0 = time.time()
    
    shape = (100, 64, 256, 256)
    dtype = np.dtype('uint16')
    arr_handle = None
    try:
        import zarr
        z_root = zarr.open(zarr_path, mode="r")
        arr_handle = z_root["0"] if "0" in z_root else z_root
        shape = tuple(arr_handle.shape)
        dtype = arr_handle.dtype
    except Exception:
        pass

    n_t = shape[0]
    print(f"\n[{zarr_idx+1}/{len(test_zarrs)}] Processing dataset: {ds_name} ({n_t} timepoints, Shape: {shape})")
    
    nodes_by_t = {}
    global_node_counter = 0

    for t in range(n_t):
        vol = get_frame(arr_handle, zarr_path, t, shape, dtype)
        tensor_in = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, Z, Y, X)

        if model:
            with torch.no_grad():
                if is_gpu:
                    with torch.amp.autocast('cuda'):
                        heat = model(tensor_in)
                else:
                    heat = model(tensor_in)
            peaks = extract_peaks_3d(heat, thresh=0.050, min_dist=3, max_peaks=500)
        else:
            vol_t = tensor_in.squeeze(0)
            peaks = extract_peaks_3d(vol_t, thresh=0.45, min_dist=3, max_peaks=500)

        t_nodes = []
        for p in peaks:
            node_id = f"{ds_name}_{global_node_counter}"
            global_node_counter += 1
            t_nodes.append({
                "id": node_id,
                "dataset": ds_name,
                "t": t,
                "z": int(p[0]),
                "y": int(p[1]),
                "x": int(p[2]),
                "pos": np.array([p[0], p[1], p[2]], dtype=np.float64)
            })
        nodes_by_t[t] = t_nodes

        if (t + 1) % 25 == 0 or t == n_t - 1:
            print(f"  Frame [{t+1}/{n_t}] done | Peak count: {len(t_nodes)} | Elapsed: {time.time()-t0:.1f}s")

    final_nodes, final_edges = track_with_gap_closing(nodes_by_t)
    total_nodes_found += len(final_nodes)
    total_edges_found += len(final_edges)

    for n in final_nodes:
        all_rows.append({
            "dataset": n["dataset"],
            "type": "node",
            "id": n["id"],
            "parent_id": "",
            "t": n["t"],
            "z": n["z"],
            "y": n["y"],
            "x": n["x"]
        })

    for src, tgt in final_edges:
        all_rows.append({
            "dataset": ds_name,
            "type": "edge",
            "id": tgt,
            "parent_id": src,
            "t": "",
            "z": "",
            "y": "",
            "x": ""
        })

    elapsed = time.time() - t0
    print(f"[{ds_name}] Complete in {elapsed:.1f}s | Nodes: {len(final_nodes)} | Edges: {len(final_edges)}")

# ----------------- 6. Generate submission.csv -----------------

out_df = pd.DataFrame(all_rows, columns=["dataset", "type", "id", "parent_id", "t", "z", "y", "x"])
if len(out_df) == 0:
    print("WARNING: Empty predictions generated! Writing valid dummy placeholder.")
    out_df = pd.DataFrame([
        {"dataset": "test_dataset", "type": "node", "id": "node_0", "parent_id": "", "t": 0, "z": 32, "y": 128, "x": 128},
        {"dataset": "test_dataset", "type": "node", "id": "node_1", "parent_id": "", "t": 1, "z": 32, "y": 128, "x": 128},
        {"dataset": "test_dataset", "type": "edge", "id": "node_1", "parent_id": "node_0", "t": "", "z": "", "y": "", "x": ""}
    ])

out_df.to_csv("submission.csv", index=False)
total_time = time.time() - start_time
csv_size_mb = os.path.getsize("submission.csv") / (1024 * 1024)

print(f"\n=======================================================")
print(f"  EXP-023 SUBMISSION COMPLETE: submission.csv GENERATED")
print(f"  Total Rows: {len(out_df)} | Total Nodes: {total_nodes_found} | Total Edges: {total_edges_found}")
print(f"  File Size: {csv_size_mb:.2f} MB | Total Time: {total_time:.1f}s")
print(f"=======================================================\n")
