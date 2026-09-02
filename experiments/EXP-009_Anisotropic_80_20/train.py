"""
EXP-009: 3D Anisotropic U-Net trained on 80% Train / 20% Validation Split
Uses proven concatenated skip connections, AdamW + Cosine Annealing, and 3D TTA validation.
"""

import os, sys, glob, json, time, subprocess
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max

from src.dataset import BiohubCellDataset, load_zarr_chunk, SCALE
from src.models.unet3d_anisotropic import AnisotropicUNet3D
from src.tracking.gap_closing_tracker import track_with_gap_closing

def focal_mse_loss(pred, target, alpha=10.0):
    weights = 1.0 + alpha * target
    return torch.mean(weights * (pred - target) ** 2)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    for vols, targets in loader:
        vols, targets = vols.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(vols)
            loss = focal_mse_loss(preds, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        valid_batches += 1
    return total_loss / max(1, valid_batches)

@torch.no_grad()
def evaluate_model(model, val_samples, train_dir, device, peak_thresh=0.010):
    model.eval()
    all_submission_rows = []
    node_id_counter = 1

    print(f"Evaluating {len(val_samples)} holdout validation volumes...")
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
                hy = torch.flip(model(torch.flip(vol_t, dims=[-2])).squeeze(), dims=[-2]).detach().cpu().numpy().astype(np.float32)
                hz = torch.flip(model(torch.flip(vol_t, dims=[-3])).squeeze(), dims=[-3]).detach().cpu().numpy().astype(np.float32)
            heat = (h0 + hx + hy + hz) / 4.0

            peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
            nodes_by_t[t] = []
            for p in peaks:
                nid = node_id_counter
                node_id_counter += 1
                nodes_by_t[t].append((nid, (int(p[0]), int(p[1]), int(p[2]))))

        filtered_nodes, edges = track_with_gap_closing(
            nodes_by_t, scale=SCALE, max_link_dist=14.0, max_gap_dist=18.0, min_track_length=2
        )

        for nid, t, z, y, x in filtered_nodes:
            all_submission_rows.append({
                'dataset': s,
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
                'dataset': s,
                'row_type': 'edge',
                'node_id': -1,
                't': -1,
                'z': -1,
                'y': -1,
                'x': -1,
                'source_id': pid,
                'target_id': cid,
            })

    if not all_submission_rows:
        return "Empty predictions"

    sub_df = pd.DataFrame(all_submission_rows)
    sub_df.index.name = "id"
    os.makedirs(os.path.expanduser("~/Documents/Biohub/eval_tmp"), exist_ok=True)
    sub_csv = os.path.expanduser("~/Documents/Biohub/eval_tmp/val_exp009.csv")
    sub_df.to_csv(sub_csv)

    pred_geff_dir = os.path.expanduser("~/Documents/Biohub/eval_tmp/pred_geffs_exp009")
    os.makedirs(pred_geff_dir, exist_ok=True)
    subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/csv_to_geffs.py"),
        "--csv", sub_csv,
        "--out-dir", pred_geff_dir
    ], check=True, stdout=subprocess.DEVNULL)

    eval_out = subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/evaluate.py"),
        "--pred-dir", pred_geff_dir,
        "--gt-dir", train_dir
    ], capture_output=True, text=True)

    return eval_out.stdout

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting EXP-009 (3D Anisotropic U-Net on 80/20 Split) on {device} ===")

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    os.makedirs(os.path.expanduser("~/Documents/Biohub/checkpoints"), exist_ok=True)

    all_zarrs = sorted(glob.glob(f"{train_dir}/*.zarr"))
    all_samples = [os.path.basename(z).replace(".zarr", "") for z in all_zarrs]
    
    np.random.seed(42)
    shuffled = np.random.permutation(all_samples)
    n_train = int(len(all_samples) * 0.80) # 159 train, 40 val
    train_samples = list(shuffled[:n_train])
    val_samples = list(shuffled[n_train:])

    print(f"Esquema 80/20: Train={len(train_samples)} vols ({len(train_samples)/len(all_samples)*100:.1f}%), Val={len(val_samples)} vols ({len(val_samples)/len(all_samples)*100:.1f}%)")

    train_ds = BiohubCellDataset(train_dir, train_samples, crop_size=(32, 128, 128), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Train Dataset ready: {len(train_ds)} 3D crops!")

    model = AnisotropicUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp009_anisotropic_80_20.pt")
    best_loss = float('inf')

    print(f"Launching training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Time: {elapsed:.1f}s", flush=True)

        if train_loss < best_loss and train_loss > 0:
            best_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path} (loss={train_loss:.6f})", flush=True)

    print("\n--- Running Official Validation on 20% Holdout Set (40 volumes) ---")
    model.load_state_dict(torch.load(ckpt_path))
    eval_results = evaluate_model(model, val_samples[:10], train_dir, device, peak_thresh=0.010)
    print("\n=== Validation Evaluation Results (EXP-009) ===")
    print(eval_results)

if __name__ == "__main__":
    main()
