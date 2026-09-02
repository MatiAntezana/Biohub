"""
EXP-029: 3D Wide Full-Volume (64, 256, 256) GroupNorm U-Net (64 Channels) on 90/10 Split
Trained on A1557-UBU RTX 4090 (24 GB VRAM).
Uses GroupNorm3d (num_groups=8) + Gradient Checkpointing for 100% numerical stability on Full Embryo 3D Volumes.
"""

import os, sys, glob, json, time, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max
import zarr

from src.dataset_full_volume import FullVolumeBiohubDataset, SCALE
from src.models.unet3d_groupnorm import GroupNormUNet3D
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
def evaluate_full_volume_gn(model, val_samples, train_dir, device, peak_thresh=0.035):
    model.eval()
    all_submission_rows = []
    node_id_counter = 1

    print(f"Evaluating {len(val_samples)} holdout validation volumes with GroupNorm Full-Volume 3D U-Net...")
    for s in val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        arr = zarr.open(z_path, mode="r")["0"]
        n_t = arr.shape[0]

        nodes_by_t = {}
        for t in range(n_t):
            vol = np.array(arr[t], dtype=np.float32)
            # Pad if needed to 64x256x256
            if vol.shape != (64, 256, 256):
                pad_z = max(0, 64 - vol.shape[0])
                pad_y = max(0, 256 - vol.shape[1])
                pad_x = max(0, 256 - vol.shape[2])
                vol = np.pad(vol, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')[:64, :256, :256]

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

    if not all_submission_rows:
        print("Empty predictions generated.")
        return

    sub_df = pd.DataFrame(all_submission_rows)
    sub_df.index.name = 'id'
    sub_path = 'val_sub_exp029.csv'
    sub_df.to_csv(sub_path)

    pred_geff_dir = 'eval_tmp/pred_geffs_exp029'
    os.makedirs(pred_geff_dir, exist_ok=True)
    subprocess.run([
        'python3', 'starter_kit/scripts/csv_to_geffs.py',
        '--csv', sub_path,
        '--out-dir', pred_geff_dir
    ], check=True, stdout=subprocess.DEVNULL)

    eval_out = subprocess.run([
        'python3', 'starter_kit/scripts/evaluate.py',
        '--pred-dir', pred_geff_dir,
        '--gt-dir', train_dir
    ], capture_output=True, text=True)
    print("\n=== Validation Evaluation Results (EXP-029 GroupNorm Full-Volume 90/10) ===")
    print(eval_out.stdout)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting EXP-029 (3D Wide Full-Volume GroupNorm 64ch on 90/10 Split) on {device} ===")

    train_dir = 'data/train'
    all_dirs = sorted([os.path.basename(p).replace('.zarr', '') for p in glob.glob(f"{train_dir}/*.zarr")])
    np.random.seed(42)
    shuffled = np.random.permutation(all_dirs)

    # 90/10 Split (179 Train, 20 Val)
    n_train = int(len(shuffled) * 0.90)
    train_samples = sorted(shuffled[:n_train].tolist())
    val_samples = sorted(shuffled[n_train:].tolist())

    print(f"Esquema 90/10: Train={len(train_samples)} vols ({len(train_samples)/len(all_dirs)*100:.1f}%), Val={len(val_samples)} vols ({len(val_samples)/len(all_dirs)*100:.1f}%)")

    train_dataset = FullVolumeBiohubDataset(
        data_dir=train_dir,
        sample_names=train_samples,
        is_train=True
    )
    print(f"Full-Volume Dataset ready: {len(train_dataset)} full 3D embryo volumes!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = GroupNormUNet3D(in_channels=1, out_channels=1, base_channels=64, gradient_checkpointing=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    ckpt_dir = '/home/alumno1/Documents/Biohub/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, 'exp029_groupnorm_full_volume_90_10.pt')

    epochs = 20
    print(f"Launching GroupNorm Full-Volume (64ch) training for {epochs} epochs on A1557...")

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
    evaluate_full_volume_gn(model, val_samples[:10], train_dir, device)

if __name__ == '__main__':
    main()
