import os, glob, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment
import blosc2

from src.dataset import BiohubCellDataset, load_zarr_chunk, SCALE
from src.models.unet3d_anisotropic import AnisotropicUNet3D

def focal_mse_loss(pred, target, alpha=10.0):
    # Weighted loss placing higher weight on cell peaks
    weights = 1.0 + alpha * target
    return torch.mean(weights * (pred - target) ** 2)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for vols, targets in loader:
        vols, targets = vols.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(vols)
            loss = focal_mse_loss(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    for vols, targets in loader:
        vols, targets = vols.to(device), targets.to(device)
        with torch.cuda.amp.autocast():
            preds = model(vols)
            loss = focal_mse_loss(preds, targets)
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def predict_sample(model, zarr_path, device, peak_thresh=0.35, max_link_dist=12.0):
    model.eval()
    with open(os.path.join(zarr_path, '0', 'zarr.json')) as f:
        meta = json.load(f)
    shape = tuple(meta['shape']) # (T, Z, Y, X)
    dtype = np.dtype(meta['data_type'])
    n_t, Z, Y, X = shape
    folder_name = os.path.basename(zarr_path).replace('.zarr', '')

    all_rows = []
    prev_centroids = {}
    node_id_counter = 1

    for t in range(n_t):
        vol = load_zarr_chunk(zarr_path, t, shape, dtype).astype(np.float32)
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
        vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

        # Predict in whole volume
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            heat = model(vol_t).squeeze().detach().cpu().numpy().astype(np.float32)

        # Find 3D peaks
        peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
        centroids = {}
        for p in peaks:
            nid = node_id_counter
            node_id_counter += 1
            centroids[nid] = (int(p[0]), int(p[1]), int(p[2]))
            all_rows.append({
                'dataset': folder_name,
                'row_type': 'node',
                'node_id': nid,
                't': t,
                'z': p[0],
                'y': p[1],
                'x': p[2],
                'source_id': -1,
                'target_id': -1,
            })

        # Match with previous frame
        if prev_centroids and len(centroids) > 0:
            prev_ids = list(prev_centroids.keys())
            curr_ids = list(centroids.keys())
            prev_coords = np.array([prev_centroids[pid] for pid in prev_ids], dtype=np.float64)
            curr_coords = np.array([centroids[cid] for cid in curr_ids], dtype=np.float64)
            prev_phys = prev_coords * SCALE
            curr_phys = curr_coords * SCALE

            dist = np.sqrt(((prev_phys[:, None] - curr_phys[None, :]) ** 2).sum(axis=2))
            row_ind, col_ind = linear_sum_assignment(dist)
            for ri, ci in zip(row_ind, col_ind):
                if dist[ri, ci] <= max_link_dist:
                    all_rows.append({
                        'dataset': folder_name,
                        'row_type': 'edge',
                        'node_id': -1,
                        't': -1,
                        'z': -1,
                        'y': -1,
                        'x': -1,
                        'source_id': prev_ids[ri],
                        'target_id': curr_ids[ci],
                    })
        prev_centroids = centroids

    return all_rows

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    os.makedirs(os.path.expanduser("~/Documents/Biohub/checkpoints"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/results"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/eval_tmp"), exist_ok=True)

    all_zarrs = sorted(glob.glob(f"{train_dir}/*.zarr"))
    all_samples = [os.path.basename(z).replace(".zarr", "") for z in all_zarrs]

    # Embryo-disjoint Split: Train on 6bba, Validate on 44b6
    train_samples = [s for s in all_samples if s.startswith("6bba")]
    val_samples = [s for s in all_samples if s.startswith("44b6")]
    print(f"Embryo-disjoint split -> Train ({len(train_samples)} samples from 6bba), Val ({len(val_samples)} samples from 44b6)")

    train_ds = BiohubCellDataset(train_dir, train_samples, crop_size=(32, 128, 128), is_train=True)
    val_ds = BiohubCellDataset(train_dir, val_samples[:10], crop_size=(32, 128, 128), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"Dataset ready: {len(train_ds)} train timepoint crops, {len(val_ds)} val crops")

    model = AnisotropicUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    epochs = 15
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp001a_unet3d_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best checkpoint: {ckpt_path} (val_loss={val_loss:.6f})", flush=True)

    print("\nTraining complete! Running end-to-end evaluation on validation set...")
    # Load best weights
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded best weights from {ckpt_path}")

    # Evaluate on a representative validation subset
    eval_val_samples = val_samples[:8]
    val_rows = []
    for s in eval_val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        val_rows.extend(predict_sample(model, z_path, device, peak_thresh=0.35, max_link_dist=12.0))

    sub_df = pd.DataFrame(val_rows)
    sub_df.index.name = 'id'
    sub_csv_path = os.path.expanduser("~/Documents/Biohub/eval_tmp/val_exp001a.csv")
    sub_df.to_csv(sub_csv_path)
    print(f"Generated {len(sub_df)} rows for {len(eval_val_samples)} validation samples. Running official metric...")

    # Convert to GEFF and run official evaluator
    import subprocess
    pred_geff_dir = os.path.expanduser("~/Documents/Biohub/eval_tmp/pred_geffs_exp001a")
    os.makedirs(pred_geff_dir, exist_ok=True)
    subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/csv_to_geffs.py"),
        "--csv", sub_csv_path,
        "--out-dir", pred_geff_dir
    ], check=True)

    eval_out = subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/evaluate.py"),
        "--pred-dir", pred_geff_dir,
        "--gt-dir", train_dir
    ], capture_output=True, text=True)

    print("\n=== Official Metric Results ===")
    print(eval_out.stdout)

    with open(os.path.expanduser("~/Documents/Biohub/results/exp001a_eval.txt"), "w") as f:
        f.write(eval_out.stdout)

if __name__ == "__main__":
    main()
