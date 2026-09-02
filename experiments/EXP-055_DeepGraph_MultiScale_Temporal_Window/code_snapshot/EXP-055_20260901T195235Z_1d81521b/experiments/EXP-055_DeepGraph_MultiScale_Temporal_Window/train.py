import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_KIT = PROJECT_ROOT / "official_starter_kit"
if str(OFFICIAL_KIT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_KIT))

from scripts.dataspec import load_dataspec, get_dataset_names
from scripts.train_unet_transformer import (
    train,
    create_model,
    _make_train_dataset,
    _make_train_dataloader,
    _train_epoch,
    compute_edge_loss,
    compute_detection_loss,
    _build_split_record,
    _resolve_split_path,
    _write_metrics_history
)
from biohub_tracking.models import TemporalUNet3D, SimpleNodeTransformer

EXPERIMENT_ID = "EXP-055"
PARENT_EXPERIMENT_ID = "EXP-051"
SEEDS = [3407, 9479]  # Dual-seed ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(EXPERIMENT_ID)

def train_fullfit_seed_with_resume(seed, role, args, run_dir, ckpt_dir):
    logger.info(f"=== Starting {EXPERIMENT_ID} ({role}, seed={seed}) with Resilient Checkpointing ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = f"exp055_fullfit_{role}_seed{seed}"
    seed_ckpt_dir = ckpt_dir / role
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    resume_ckpt_path = seed_ckpt_dir / "latest_checkpoint.pth"
    final_ckpt_path = seed_ckpt_dir / "training_checkpoint.pth"
    
    if final_ckpt_path.exists():
        logger.info(f"Final checkpoint already exists at {final_ckpt_path}! Skipping seed training.")
        return

    # Create Multi-Scale Temporal Graph Model
    logger.info("Initializing Multi-Scale TemporalUNet3D (downsample=(1,2,2), window_size=5, d_model=192, nhead=6)")
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=32,
        layers=(32, 64, 128),
        downsample=(1, 2, 2),
        upsample=(1, 2, 2)
    )
    model = SimpleNodeTransformer(
        unet=unet,
        d_model=192,
        nhead=6,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.1,
        pool_kernel_um=5.0,
        det_loss_weight=1.5
    )
    model = model.to(device)

    dataspec_path = PROJECT_ROOT / "official_starter_kit" / "data" / "dataspec_all.yaml"
    dataspec = load_dataspec(str(dataspec_path))
    
    split_record = _build_split_record(dataspec_path, method, train_fraction=1.0)
    split_record_path = run_dir / "full_fit_split.json"
    with open(split_record_path, "w") as f:
        json.dump(split_record, f, indent=2)

    train_dataset = _make_train_dataset(
        dataspec=dataspec,
        split_record=split_record,
        window_size=5,
        target_shape=(64, 256, 256),
        patch_size=None,
        aug_cfg={"flip_prob": 0.5, "rot_prob": 0.5, "gamma_prob": 0.3}
    )
    train_loader = _make_train_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 0
    history = []
    best_loss = float("inf")

    if resume_ckpt_path.exists():
        logger.info(f"Detected checkpoint {resume_ckpt_path}! Resuming training state...")
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        history = checkpoint.get("history", [])
        best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Successfully resumed at Epoch {start_epoch} (Best Loss: {best_loss:.6f})")

    t0_train = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0_epoch = time.time()
        model.train()
        epoch_edge_loss = 0.0
        epoch_det_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_iters and batch_idx >= args.max_iters:
                break
            
            imgs = batch["imgs"].to(device)  # (B, T, 1, Z, Y, X)
            node_graphs = batch.get("node_graphs")

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                feats, det_logits = model(imgs)
                det_loss = compute_detection_loss(det_logits, batch.get("heatmaps").to(device)) if "heatmaps" in batch else torch.tensor(0.0, device=device)
                edge_loss = compute_edge_loss(model, feats, node_graphs, device=device)
                loss = edge_loss + 1.5 * det_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_edge_loss += edge_loss.item()
            epoch_det_loss += det_loss.item() if isinstance(det_loss, torch.Tensor) else det_loss
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0_epoch
        avg_edge_loss = epoch_edge_loss / max(1, n_batches)
        avg_det_loss = epoch_det_loss / max(1, n_batches)
        total_epoch_loss = avg_edge_loss + 1.5 * avg_det_loss

        epoch_metric = {
            "epoch": epoch,
            "edge_loss": avg_edge_loss,
            "detection_loss": avg_det_loss,
            "total_loss": total_epoch_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": epoch_time
        }
        history.append(epoch_metric)
        logger.info(f"[{role} Seed {seed}] Epoch {epoch:2d}/{args.epochs} | Edge Loss: {avg_edge_loss:.6f} | Det Loss: {avg_det_loss:.6f} | Time: {epoch_time:.1f}s")

        # Atomic Resilient Checkpoint Save
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_loss": min(best_loss, total_epoch_loss),
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "seed": seed,
                "role": role,
                "downsample": [1, 2, 2],
                "window_size": 5,
                "d_model": 192,
                "nhead": 6
            }
        }
        tmp_ckpt = seed_ckpt_dir / "latest_checkpoint.pth.tmp"
        torch.save(ckpt_payload, tmp_ckpt)
        os.replace(tmp_ckpt, resume_ckpt_path)

        # Write metrics history
        _write_metrics_history(history, run_dir / f"metrics_{role}.json")

        # Write live heartbeat
        heartbeat = {
            "experiment_id": EXPERIMENT_ID,
            "role": role,
            "seed": seed,
            "epoch": epoch,
            "total_epochs": args.epochs,
            "latest_loss": total_epoch_loss,
            "timestamp": time.time()
        }
        with open(run_dir / "HEARTBEAT.json", "w") as f:
            json.dump(heartbeat, f, indent=2)

    # Save final checkpoint
    torch.save(ckpt_payload, final_ckpt_path)
    logger.info(f"Saved final checkpoint to {final_ckpt_path}")
    
    # Save config
    with open(seed_ckpt_dir / "config.json", "w") as f:
        json.dump(ckpt_payload["config"], f, indent=2)

def main():
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} Training Entrypoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=750)
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "experiments" / "EXP-055_DeepGraph_MultiScale_Temporal_Window" / "artifacts" / f"EXP-055_local_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir.parent.parent / "checkpoints" / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Seed 1: Primary (3407)
    train_fullfit_seed_with_resume(SEEDS[0], "primary", args, run_dir, ckpt_dir)

    # Seed 2: Secondary (9479)
    train_fullfit_seed_with_resume(SEEDS[1], "secondary", args, run_dir, ckpt_dir)

    # Final Receipt
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "completed_at": time.time(),
        "status": "SUCCEEDED",
        "primary_checkpoint": str(ckpt_dir / "primary" / "training_checkpoint.pth"),
        "secondary_checkpoint": str(ckpt_dir / "secondary" / "training_checkpoint.pth")
    }
    with open(run_dir / "TRAINING_COMPLETE.json", "w") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"=== {EXPERIMENT_ID} FULL TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()
