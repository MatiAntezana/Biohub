"""
EXP-028: 3D Wide Attention U-Net (64 Base Channels + Attention Gates) on 90/10 Split (179 Train vols)
Trained on A1560-UBU RTX 4090.
Applies 3D Spatial Attention Gates to suppress background noise and focus on dividing blastomeres.
"""

import os, sys, glob, json, time, subprocess
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max

from src.dataset import BiohubCellDataset, load_zarr_chunk, SCALE
from src.models.attention_unet3d import AttentionUNet3D
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
def evaluate_attention(model, val_samples, train_dir, device, peak_thresh=0.035):
    model.eval()
    all_submission_rows = []
    node_id_counter = 1

    print(f"Evaluating {len(val_samples)} holdout validation volumes (Wide Attention UNet 90/10)...")
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

    sub_df = pd.DataFrame(all_submission_rows)
    sub_df.index.name = 'id'
    sub_path = 'val_sub_exp028.csv'
    sub_df.to_csv(sub_path)

    cmd = ['python3', '-m', 'src.metrics.evaluate', sub_path, train_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n=== Validation Evaluation Results (EXP-028 3D Wide Attention UNet 90/10) ===")
    print(res.stdout)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting EXP-028 (3D Wide Attention U-Net 64ch on 90/10 Split) on {device} ===")

    train_dir = 'data/train'
    all_dirs = sorted([os.path.basename(p).replace('.zarr', '') for p in glob.glob(f"{train_dir}/*.zarr")])
    np.random.seed(42)
    shuffled = np.random.permutation(all_dirs)

    # 90/10 Split
    n_train = int(len(shuffled) * 0.90)
    train_samples = sorted(shuffled[:n_train].tolist())
    val_samples = sorted(shuffled[n_train:].tolist())

    print(f"Esquema 90/10: Train={len(train_samples)} vols ({len(train_samples)/len(all_dirs)*100:.1f}%), Val={len(val_samples)} vols ({len(val_samples)/len(all_dirs)*100:.1f}%)")

    train_dataset = BiohubCellDataset(
        data_dir=train_dir,
        sample_names=train_samples,
        crop_size=(32, 64, 64),
        is_train=True
    )
    print(f"Train Dataset ready: {len(train_dataset)} 3D crops!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True
    )

    model = AttentionUNet3D(in_channels=1, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    ckpt_dir = '/home/alumno1/Documents/Biohub/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, 'exp028_wide_attention_unet3d_90_10.pt')

    epochs = 20
    print(f"Launching 3D Wide Attention U-Net (64ch) training for {epochs} epochs on A1560...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Time: {elapsed:.1f}s")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best checkpoint: {best_model_path} (loss={train_loss:.6f})")

    print("\n--- Running Official Validation on 10% Holdout Set (20 volumes) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    evaluate_attention(model, val_samples[:10], train_dir, device)

if __name__ == '__main__':
    main()
