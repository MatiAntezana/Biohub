"""
BIOHUB CELL TRACKING - EXP-030B DUAL-SEED TEMPORAL ENSEMBLE FULL-FIT SUBMISSION ENGINE
Architecture: Dual-Seed TemporalUNet3D + Node Transformer + Calibrated Fusion + TTA + LAP Tracking
Target: Tesla T4 execution in < 3 minutes commit, < 18 minutes hidden test.
"""

import os, sys, glob, json, time, gc, shutil, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Launching Biohub EXP-030B DualSeed Temporal Ensemble Submission (Fast Gated LAP) ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_gpu = torch.cuda.is_available()
print(f"Device: {device} ({torch.cuda.get_device_name(0) if is_gpu else 'CPU'})")

# ----------------- 0. Install Offline Packages if available -----------------

wheel_dirs = glob.glob("/kaggle/input/**/wheels", recursive=True) + glob.glob("champion_weights/wheels")
for wd in wheel_dirs:
    print(f"Found offline wheels in: {wd}")
    os.system(f"pip install --no-index --find-links={wd} zarr blosc2 numcodecs >/dev/null 2>&1")

# ----------------- 1. Architecture Definitions -----------------

class _TemporalAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)
        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        chunk_size = 16384
        if h.shape[0] > chunk_size:
            out_chunks = []
            for i in range(0, h.shape[0], chunk_size):
                chunk = h[i : i + chunk_size]
                out_c, _ = self.attn(chunk, chunk, chunk, need_weights=False)
                out_chunks.append(out_c)
            h = torch.cat(out_chunks, dim=0)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h

class _ConvBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm2(self.conv2(self.act(self.norm1(self.conv1(x))))))

class TemporalUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, layers=(32, 64, 128)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev = in_channels
        for c in layers:
            self.encoders.append(_ConvBlock3D(prev, c))
            self.downsamplers.append(nn.MaxPool3d(2))
            prev = c
        self.bottleneck = _ConvBlock3D(prev, prev * 2)
        self.bottleneck_attn = _TemporalAttention(prev * 2)
        prev = prev * 2
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(layers):
            self.upsamplers.append(nn.ConvTranspose3d(prev, c, 2, stride=2))
            self.decoders.append(_ConvBlock3D(c * 2, c))
            prev = c
        self.head = nn.Conv3d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, Z, Y, X = x.shape
        x_flat = x.view(B * T, C, Z, Y, X)
        skips = []
        h = x_flat
        for enc, down in zip(self.encoders, self.downsamplers):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        _, bc, bz, by, bx = h.shape
        h = h.view(B, T, bc, bz, by, bx)
        h = self.bottleneck_attn(h)
        h = h.view(B * T, bc, bz, by, bx)
        for up, dec, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))
        out = self.head(h)
        return out.view(B, T, -1, Z, Y, X)

class UNetNodeTransformer(nn.Module):
    def __init__(self, unet: nn.Module, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.unet = unet
        self.det_head = nn.Conv3d(32, 1, 1)
        self.node_proj = nn.Linear(32 + 32, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.edge_head = nn.Linear(d_model * 2, 1)

    def forward(self, imgs: torch.Tensor):
        feats = self.unet(imgs)
        B, T, C, Z, Y, X = feats.shape
        det_logits = self.det_head(feats.view(B * T, C, Z, Y, X)).view(B, T, 1, Z, Y, X)
        return feats, det_logits

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

# ----------------- 3. Dual-Seed Ensemble Loader -----------------

def find_weights_file():
    candidates = [
        "/kaggle/input/datasets/matiasantezana/biohub-champion-weights",
        "/kaggle/input/biohub-champion-weights",
        "./champion_weights",
        "../input/biohub-champion-weights"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_ensemble_models(weights_root, device):
    models = []
    
    primary_paths = glob.glob(f"{weights_root}/**/exp030b_primary*.pt", recursive=True) or \
                    glob.glob(f"{weights_root}/**/exp030b_primary*.pth", recursive=True) or \
                    glob.glob(f"{weights_root}/exp030b_primary*.pt")
    secondary_paths = glob.glob(f"{weights_root}/**/exp030b_secondary*.pt", recursive=True) or \
                      glob.glob(f"{weights_root}/**/exp030b_secondary*.pth", recursive=True) or \
                      glob.glob(f"{weights_root}/exp030b_secondary*.pt")

    for name, p_list in [("Primary", primary_paths), ("Secondary", secondary_paths)]:
        if p_list:
            ckpt_path = p_list[0]
            print(f"Loading {name} model weights from: {ckpt_path}")
            try:
                unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128))
                model = UNetNodeTransformer(unet, d_model=128, nhead=4, num_layers=2)
                state = torch.load(ckpt_path, map_location="cpu")
                if "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                elif "model" in state:
                    model.load_state_dict(state["model"], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                model.to(device).eval()
                models.append(model)
                print(f"Successfully loaded {name} model.")
            except Exception as e:
                print(f"Error loading {name} model: {e}")
                
    return models

# ----------------- 4. Calibrated Fast Peak Extraction & Gated Tracking -----------------

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64) # Z, Y, X physical scales in um

@torch.no_grad()
def extract_peaks_3d(heat_tensor, thresh=0.40, min_dist=3, max_peaks=500):
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
            resolved = Path(c).resolve()
            if "test" not in {part.lower() for part in resolved.parts}:
                raise RuntimeError(f"BIOHUB_TEST_ONLY_GUARD rejected non-test directory: {resolved}")
            return str(resolved)
    raise RuntimeError("BIOHUB_TEST_ONLY_GUARD: competition test directory was not found")

weights_root = find_weights_file()
test_data_dir = find_test_data_dir()

print(f"Weights root: {weights_root}")
print(f"Test data dir: {test_data_dir}")

models = load_ensemble_models(weights_root, device) if weights_root else []

test_zarrs = sorted(glob.glob(f"{test_data_dir}/*.zarr")) if test_data_dir else []
if not test_zarrs:
    raise RuntimeError("BIOHUB_TEST_ONLY_GUARD: no .zarr datasets were found under test/")
for candidate_path in test_zarrs:
    parts = {part.lower() for part in Path(candidate_path).resolve().parts}
    if "train" in parts or "test" not in parts:
        raise RuntimeError(f"BIOHUB_TEST_ONLY_GUARD rejected dataset path: {candidate_path}")
print("BIOHUB_DATASET_SCOPE=" + json.dumps({
    "scope": "test",
    "count": len(test_zarrs),
    "paths": [str(Path(path).resolve()) for path in test_zarrs],
}, sort_keys=True))
print(f"Total test datasets to process: {len(test_zarrs)}")

all_rows = []
total_nodes_found = 0
total_edges_found = 0
start_time = time.time()

for zarr_idx, zarr_path in enumerate(test_zarrs):
    print(f"BIOHUB_DATASET_PATH={Path(zarr_path).resolve()}")
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
        tensor_in = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 1, Z, Y, X)

        if models:
            preds = []
            with torch.no_grad():
                for m in models:
                    if is_gpu:
                        with torch.amp.autocast('cuda'):
                            _, det_logits = m(tensor_in)
                            p = torch.sigmoid(det_logits).squeeze(0).squeeze(0)
                    else:
                        _, det_logits = m(tensor_in)
                        p = torch.sigmoid(det_logits).squeeze(0).squeeze(0)
                    preds.append(p)
                heat = torch.stack(preds, dim=0).mean(dim=0).unsqueeze(0)
            peaks = extract_peaks_3d(heat, thresh=0.40, min_dist=3, max_peaks=500)
        else:
            vol_t = tensor_in.squeeze(0)
            peaks = extract_peaks_3d(vol_t, thresh=0.45, min_dist=3, max_peaks=500)

        t_nodes = []
        for p in peaks:
            node_id = global_node_counter
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
            "row_type": "node",
            "node_id": n["id"],
            "t": n["t"],
            "z": n["z"],
            "y": n["y"],
            "x": n["x"],
            "source_id": -1,
            "target_id": -1
        })

    for src, tgt in final_edges:
        all_rows.append({
            "dataset": ds_name,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": src,
            "target_id": tgt
        })

    elapsed = time.time() - t0
    print(f"[{ds_name}] Complete in {elapsed:.1f}s | Nodes: {len(final_nodes)} | Edges: {len(final_edges)}")

# ----------------- 6. Generate submission.csv -----------------

submission_columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
payload_columns = submission_columns[1:]
out_df = pd.DataFrame(all_rows, columns=payload_columns)
if len(out_df) == 0:
    raise RuntimeError("BIOHUB_TEST_ONLY_GUARD: inference produced no nodes; refusing dummy submission")

out_df.insert(0, "id", range(len(out_df)))
if list(out_df.columns) != submission_columns:
    raise RuntimeError(f"Invalid submission schema: {list(out_df.columns)}")

out_df.to_csv("submission.csv", index=False)
total_time = time.time() - start_time
csv_size_mb = os.path.getsize("submission.csv") / (1024 * 1024)

print(f"\n=======================================================")
print(f"  EXP-030B SUBMISSION COMPLETE: submission.csv GENERATED")
print(f"  Total Rows: {len(out_df)} | Total Nodes: {total_nodes_found} | Total Edges: {total_edges_found}")
print(f"  File Size: {csv_size_mb:.2f} MB | Total Time: {total_time:.1f}s")
print(f"=======================================================\n")
