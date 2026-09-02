"""
EXP-005: Full-Fit 3D Anisotropic U-Net Training on 100% of Training Data (199 Datasets)
Combines all developmental stages of embryos 6bba and 44b6.
"""

import os, glob, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BiohubCellDataset
from src.models.unet3d_anisotropic import AnisotropicUNet3D

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
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting EXP-005 Full-Fit Training on {device} ({torch.cuda.get_device_name(0)}) ===")

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    os.makedirs(os.path.expanduser("~/Documents/Biohub/checkpoints"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/results"), exist_ok=True)

    all_zarrs = sorted(glob.glob(f"{train_dir}/*.zarr"))
    all_samples = [os.path.basename(z).replace(".zarr", "") for z in all_zarrs]
    print(f"Training on 100% of train set: {len(all_samples)} volumes across all embryos!")

    # Full Dataset
    train_ds = BiohubCellDataset(train_dir, all_samples, crop_size=(32, 128, 128), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Full dataset ready: {len(train_ds)} total 3D crops!")

    model = AnisotropicUNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    
    # Initialize from best validation weights to accelerate convergence
    init_ckpt = os.path.expanduser("~/Documents/Biohub/checkpoints/exp001a_unet3d_best.pt")
    if os.path.exists(init_ckpt):
        model.load_state_dict(torch.load(init_ckpt))
        print(f"Initialized weights from champion checkpoint: {init_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp005_fullfit_best.pt")
    best_loss = float('inf')

    print(f"Launching Full-Fit training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Time: {elapsed:.1f}s", flush=True)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved new best full-fit checkpoint: {ckpt_path} (loss={train_loss:.6f})", flush=True)

    print("\nEXP-005 Full-Fit Training completed successfully!")

if __name__ == "__main__":
    main()
