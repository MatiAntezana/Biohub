#!/usr/bin/env python3
"""EXP-058: DeepGraph Mitosis-Aware HighRes Temporal T5.

Combines high-resolution spatial grid (1, 2, 2), 5-frame temporal horizon,
anisotropic Z-pooling at 3.75 um, deep node transformer (d_model=192, nhead=6),
and boosted detection loss (weight=1.75, neg_weight=0.018) with cosine schedule.

Both TemporalUNet3D + Transformer seeds train on every paired Biohub training
dataset. There is deliberately no validation holdout and therefore no local
validation score for this run. Execute only on a remote CUDA machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if sys.platform == "darwin":
    sys.path.insert(0, str(PROJECT_ROOT / "infra"))
    from aienv import ensure_aienv
    ensure_aienv()

from src.strategies.experiment_runtime import (
    require_remote_cuda,
    seed_everything,
    sha256_file,
    write_full_fit_split,
    write_manifest,
)


EXPERIMENT_ID = "EXP-058"
PARENT_EXPERIMENT_ID = "EXP-055"
SEEDS = (3407, 9479)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--starter-kit", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=750)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume-primary", type=Path, default=None)
    parser.add_argument("--resume-secondary", type=Path, default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_plan(args: argparse.Namespace) -> dict:
    project_root = args.project_root.expanduser().resolve()
    data_dir = (args.data_dir or project_root / "data" / "train").expanduser().resolve()
    starter_kit = (args.starter_kit or project_root / "official_starter_kit").expanduser().resolve()
    experiment_dir = Path(__file__).resolve().parent
    artifact_dir = (args.run_dir or experiment_dir / "artifacts" / "dry-run").expanduser().resolve()
    return {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "starter_kit": str(starter_kit),
        "artifact_dir": str(artifact_dir),
        "experiment_dir": str(experiment_dir),
        "run_id": artifact_dir.name,
        "epochs_per_seed": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_iters_per_epoch": args.max_iters,
        "seeds": list(SEEDS),
        "split_policy": "full-fit: 100% train / 0% validation",
        "validation": "none",
        "checkpoint_selection": "last_epoch",
        "window_size": 5,
        "hidden_dim": 192,
        "n_heads": 6,
        "dropout": 0.1,
        "downsample_zyx": [1, 2, 2],
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "det_loss_weight": 1.75,
        "det_negative_weight": 0.018,
        "pool_kernel_um": 3.75,
        "ensemble": {
            "detection_secondary_weight": 0.475,
            "edge_secondary_weight": 0.15,
            "retention_minimum": 0.90,
            "primary_margin_limit": 0.35,
            "reverse_weight": 0.30,
            "std_ratio_minimum": 0.50,
            "std_ratio_maximum": 2.00,
        },
    }


def main() -> None:
    args = parse_args()
    plan = resolved_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    require_remote_cuda()
    project_root = Path(plan["project_root"])
    data_dir = Path(plan["data_dir"])
    starter_kit = Path(plan["starter_kit"])
    experiment_dir = Path(plan["experiment_dir"])
    artifact_dir = Path(plan["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = starter_kit / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from train_unet_transformer import train  # type: ignore

    splits_path = artifact_dir / "full_fit_splits.json"
    paired_datasets = write_full_fit_split(data_dir, splits_path)
    plan["train_dataset_count"] = len(paired_datasets)
    print(json.dumps(plan, indent=2))
    print(f"Paired datasets verified for full fit ({len(paired_datasets)}): {paired_datasets[:5]} ...", flush=True)

    running_path = artifact_dir / "RUNNING.json"
    complete_path = artifact_dir / "TRAINING_COMPLETE.json"
    failed_path = artifact_dir / "FAILED.json"
    receipt = {**plan, "started_at": utc_now(), "status": "RUNNING", "checkpoints": []}
    write_manifest(running_path, receipt)

    try:
        for role, seed in zip(("primary", "secondary"), SEEDS, strict=True):
            method = f"exp058_fullfit_{role}_seed{seed}"
            checkpoint_dir = experiment_dir / "checkpoints" / plan["run_id"] / role
            metrics_path = artifact_dir / f"metrics_{role}.json"
            resume_from = args.resume_primary if role == "primary" else args.resume_secondary
            if resume_from is not None and not resume_from.is_absolute():
                candidate = (experiment_dir / resume_from).resolve()
                if not candidate.is_file():
                    candidate = (project_root / resume_from).resolve()
                resume_from = candidate
            seed_everything(seed)
            print(f"\n=== {EXPERIMENT_ID} {role.upper()} | seed={seed} ===", flush=True)
            train(
                data_dir=data_dir,
                fold=0,
                splits_file=splits_path,
                method=method,
                n_epochs=args.epochs,
                lr=1e-3,
                weight_decay=0.0002,
                lr_scheduler_type="cosine",
                min_lr=5e-5,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                unet_out_channels=32,
                unet_layers=[32, 64, 128],
                downsample=(1, 2, 2),
                det_loss_weight=1.75,
                det_neg_weight=0.018,
                max_iters=args.max_iters,
                seed=seed,
                window_size=5,
                pool_kernel_um=3.75,
                data_parallel=False,
                full_fit=True,
                output_dir=checkpoint_dir,
                progress=not args.compact,
                metrics_path=metrics_path,
                checkpoint_bundle=True,
                resume_from=resume_from,
                hidden_dim=192,
                n_heads=6,
                dropout=0.1,
            )
            checkpoint = checkpoint_dir / "training_checkpoint.pth"
            config_path = checkpoint.parent / "config.json"
            if not checkpoint.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"training finished without expected checkpoint: {checkpoint}")
            receipt["checkpoints"].append(
                {
                    "role": role,
                    "seed": seed,
                    "method": method,
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "config": str(config_path),
                }
            )

        metrics = {}
        for role in ("primary", "secondary"):
            role_metrics = artifact_dir / f"metrics_{role}.json"
            if role_metrics.is_file():
                payload = json.loads(role_metrics.read_text(encoding="utf-8"))
                metrics[role] = payload[-1] if payload else {}
        receipt["status"] = "COMPLETE"
        receipt["completed_at"] = utc_now()
        receipt["metrics"] = metrics
        write_manifest(complete_path, receipt)
        if running_path.is_file():
            running_path.unlink()
        print(f"\nTraining finished successfully. Manifest: {complete_path}", flush=True)
    except Exception as error:
        receipt["status"] = "FAILED"
        receipt["failed_at"] = utc_now()
        receipt["error"] = str(error)
        receipt["traceback"] = traceback.format_exc()
        write_manifest(failed_path, receipt)
        if running_path.is_file():
            running_path.unlink()
        print(f"\nTraining failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
