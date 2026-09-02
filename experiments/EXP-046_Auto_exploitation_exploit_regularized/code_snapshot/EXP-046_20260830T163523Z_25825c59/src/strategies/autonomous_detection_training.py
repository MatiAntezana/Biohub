"""Normalized full-fit trainer used by policy-generated detection variants.

The module deliberately contains no local fallback: it refuses to train
without CUDA and writes every checkpoint/receipt inside the experiment that
owns the run.  Variant-specific choices live in ``variant.json`` so generated
``train.py`` files remain tiny and auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.strategies.experiment_runtime import (
    dataset_stems,
    require_remote_cuda,
    seed_everything,
    sha256_file,
    write_full_fit_split,
    write_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model(architecture: str, base_channels: int):
    if architecture == "anisotropic_unet3d":
        from src.models.unet3d_anisotropic import AnisotropicUNet3D

        return AnisotropicUNet3D(1, 1, base_channels, gradient_checkpointing=True)
    if architecture == "resunet3d":
        from src.models.resnet_unet3d import ResUNet3D

        return ResUNet3D(1, 1, base_channels)
    if architecture == "groupnorm_unet3d":
        from src.models.unet3d_groupnorm import GroupNormUNet3D

        return GroupNormUNet3D(1, 1, base_channels, gradient_checkpointing=True)
    if architecture == "attention_unet3d":
        from src.models.attention_unet3d import AttentionUNet3D

        return AttentionUNet3D(1, 1, base_channels)
    if architecture == "convnext_unet3d":
        from src.models.convnext_unet3d import ConvNeXtUNet3D

        return ConvNeXtUNet3D(1, 1, base_channels)
    raise ValueError(f"arquitectura no permitida por el trainer autónomo: {architecture}")


def _focal_mse(pred, target, alpha: float):
    import torch

    return torch.mean((1.0 + alpha * target) * (pred - target) ** 2)


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant-file", type=Path, required=True)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = _args(argv)
    variant_path = args.variant_file.expanduser().resolve()
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    experiment_dir = variant_path.parent
    project_root = args.project_root.expanduser().resolve()
    data_dir = (args.data_dir or project_root / "data" / "train").expanduser().resolve()
    training = variant["training"]
    epochs = int(args.epochs or training["epochs"])
    batch_size = int(args.batch_size or training["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    max_batches = int(args.max_batches_per_epoch or training["max_batches_per_epoch"])
    run_dir = (args.run_dir or experiment_dir / "artifacts" / "dry-run").expanduser().resolve()
    plan = {
        "experiment_id": variant["experiment_id"],
        "proposal_id": variant["proposal_id"],
        "parent_experiment_id": variant.get("parent_experiment_id"),
        "lane": variant["lane"],
        "recipe_id": variant["recipe_id"],
        "architecture": training["architecture"],
        "base_channels": int(training["base_channels"]),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": workers,
        "max_batches_per_epoch": max_batches,
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "focal_alpha": float(training["focal_alpha"]),
        "crop_size": list(training["crop_size"]),
        "seed": int(training["seed"]),
        "data_policy": "full-fit: 100% train / 0% validation",
        "validation": "N/A — full-fit sin validación",
        "run_id": run_dir.name,
    }
    print(json.dumps({"event": "plan", **plan}, sort_keys=True), flush=True)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", "experiment_id": variant["experiment_id"]}), flush=True)
        return
    if args.run_dir is None:
        raise ValueError("--run-dir es obligatorio fuera de --dry-run; usar experimentctl run")
    try:
        run_dir.relative_to(experiment_dir / "artifacts")
    except ValueError as error:
        raise ValueError("run-dir debe vivir dentro de artifacts/ del experimento") from error

    import torch
    from torch.utils.data import DataLoader

    from src.dataset import BiohubCellDataset

    device = require_remote_cuda()
    seed_everything(plan["seed"])
    stems = dataset_stems(data_dir)
    write_full_fit_split(data_dir, run_dir / "full_fit_split.json")
    dataset = BiohubCellDataset(
        str(data_dir), stems, crop_size=tuple(plan["crop_size"]), is_train=True
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    model = _model(plan["architecture"], plan["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=plan["learning_rate"], weight_decay=plan["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training.get("eta_min", 1e-5))
    )
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = experiment_dir / "checkpoints" / plan["run_id"] / "training_checkpoint.pth"
    start_epoch = 1
    best_train_loss = float("inf")
    history: list[dict[str, Any]] = []
    if args.resume_from:
        bundle = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(bundle.get("model_state_dict", bundle.get("model")))
        optimizer.load_state_dict(bundle.get("optimizer_state_dict", bundle.get("optimizer")))
        scheduler.load_state_dict(bundle["scheduler"])
        scaler.load_state_dict(bundle.get("scaler_state_dict", bundle.get("scaler")))
        start_epoch = int(bundle["epoch"]) + 1
        best_train_loss = float(bundle.get("best_train_loss", best_train_loss))
        history = list(bundle.get("history", []))

    receipt = {
        **plan,
        "status": "RUNNING",
        "started_at": utc_now(),
        "dataset_count": len(stems),
        "dataset_samples": len(dataset),
        "checkpoint": str(checkpoint),
    }
    write_manifest(run_dir / "RUNNING.json", receipt)
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total = 0.0
            batches = 0
            started = time.monotonic()
            for volumes, targets in loader:
                if batches >= max_batches:
                    break
                volumes = volumes.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    prediction = model(volumes)
                    loss = _focal_mse(prediction, targets, plan["focal_alpha"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss no finita en epoch={epoch}, batch={batches}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach().cpu())
                batches += 1
            scheduler.step()
            mean_loss = total / max(1, batches)
            best_train_loss = min(best_train_loss, mean_loss)
            row = {
                "epoch": epoch,
                "train_loss_diagnostic": mean_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batches": batches,
                "seconds": round(time.monotonic() - started, 3),
            }
            history.append(row)
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "format_version": 1,
                    "experiment_id": variant["experiment_id"],
                    "proposal_id": variant["proposal_id"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "best_train_loss": best_train_loss,
                    "history": history,
                    "rng": _rng_state(),
                    "variant": variant,
                },
                checkpoint,
            )
            print(json.dumps({"event": "epoch", **row}, sort_keys=True), flush=True)

        final = {
            **receipt,
            "status": "COMPLETE",
            "completed_at": utc_now(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics": {
                "train_loss_diagnostic": history[-1]["train_loss_diagnostic"],
                "best_train_loss_diagnostic": best_train_loss,
                "validation_score": None,
                "kaggle_public_score": None,
            },
        }
        write_manifest(run_dir / "TRAINING_COMPLETE.json", final)
        print(json.dumps({"event": "complete", "checkpoint": str(checkpoint)}), flush=True)
    except Exception as error:
        write_manifest(
            run_dir / "FAILED.json",
            {**receipt, "status": "FAILED", "failed_at": utc_now(), "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
