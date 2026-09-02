"""
EXP-007: 3D Residual U-Net with Bottleneck Self-Attention & Deep Supervision
Trained on 100% of data (199 volumes) on RTX 4090.
"""

import os, glob, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BiohubCellDataset
from src.models.resnet_unet3d import ResNetUNet3D

def focal_mse_loss(pred, target, alpha=10.0):
    weights = 1.0 + alpha * target
    return torch.mean(weights * (pred - target) ** 2)

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
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
    return total_loss / max(1, len(loader))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting EXP-007 (3D ResNet-UNet + Self-Attention) Training on {device} ===")

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    os.makedirs(os.path.expanduser("~/Documents/Biohub/checkpoints"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/results"), exist_ok=True)

    all_zarrs = sorted(glob.glob(f"{train_dir}/*.zarr"))
    all_samples = [os.path.basename(z).replace(".zarr", "") for z in all_zarrs]
    print(f"Training ResNet-UNet 3D on 100% data: {len(all_samples)} volumes!")

    train_ds = BiohubCellDataset(train_dir, all_samples, crop_size=(32, 128, 128), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset ready: {len(train_ds)} 3D crops!")

    model = ResNetUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp007_resnet_best.pt")
    best_loss = float('inf')

    print(f"Launching 3D ResNet training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Time: {elapsed:.1f}s", flush=True)

        if train_loss < best_loss and train_loss > 0:
            best_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best 3D ResNet checkpoint: {ckpt_path} (loss={train_loss:.6f})", flush=True)

    print("\nEXP-007 Training completed successfully!")

if __name__ == "__main__":
    main()
