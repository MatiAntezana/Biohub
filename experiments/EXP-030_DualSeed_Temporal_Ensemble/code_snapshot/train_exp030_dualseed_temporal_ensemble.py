#!/usr/bin/env python3
"""EXP-030: train two independent TemporalUNet3D + Transformer seeds.

This is the training half of the independent dual-seed/TTA/ILP strategy.  It
must be launched only on a remote CUDA machine with the complete Biohub train
dataset.  The script writes explicit RUNNING/COMPLETE/FAILED receipts so a
controller can monitor it without parsing a terminal session.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.strategies.experiment_runtime import (
    require_remote_cuda,
    seed_everything,
    sha256_file,
    write_embryo_disjoint_splits,
    write_manifest,
)


EXPERIMENT_ID = "EXP-030"
SEEDS = (3407, 9479)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--starter-kit", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_plan(args: argparse.Namespace) -> dict:
    project_root = args.project_root.expanduser().resolve()
    data_dir = (args.data_dir or project_root / "data" / "train").expanduser().resolve()
    starter_kit = (args.starter_kit or project_root / "official_starter_kit").expanduser().resolve()
    artifact_dir = project_root / "experiment_artifacts" / "EXP-030"
    return {
        "experiment_id": EXPERIMENT_ID,
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "starter_kit": str(starter_kit),
        "artifact_dir": str(artifact_dir),
        "fold": args.fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_iters": args.max_iters,
        "seeds": list(SEEDS),
        "split_policy": "leave-one-embryo-prefix-out",
        "window_size": 3,
        "downsample_zyx": [1, 4, 4],
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "det_loss_weight": 1.0,
        "det_negative_weight": 0.02,
        "pool_kernel_um": 5.0,
        "ensemble": {
            "detection_secondary_weight": 0.475,
            "edge_secondary_weight": 0.15,
            "retention_minimum": 0.90,
            "primary_margin_limit": 0.35,
            "reverse_weight": 0.30,
            "std_ratio_minimum": 0.5,
            "std_ratio_maximum": 2.0,
        },
    }


def main() -> None:
    args = parse_args()
    plan = resolved_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN: configuration resolved; no dataset, CUDA or training was touched.")
        return

    require_remote_cuda()
    project_root = Path(plan["project_root"])
    data_dir = Path(plan["data_dir"])
    starter_kit = Path(plan["starter_kit"])
    artifact_dir = Path(plan["artifact_dir"])
    scripts_dir = starter_kit / "scripts"
    source_dir = starter_kit / "src"
    if not scripts_dir.is_dir() or not source_dir.is_dir():
        raise FileNotFoundError(f"Incomplete starter kit under {starter_kit}")

    splits_path = artifact_dir / "embryo_disjoint_splits.json"
    folds = write_embryo_disjoint_splits(data_dir, splits_path)
    if not 0 <= args.fold < len(folds):
        raise ValueError(f"fold {args.fold} outside available range 0..{len(folds)-1}")

    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    from train_unet_transformer import train  # noqa: PLC0415

    running_path = artifact_dir / "RUNNING.json"
    complete_path = artifact_dir / "TRAINING_COMPLETE.json"
    failed_path = artifact_dir / "FAILED.json"
    receipt = {**plan, "started_at": utc_now(), "status": "RUNNING", "checkpoints": []}
    write_manifest(running_path, receipt)

    try:
        for role, seed in zip(("primary", "secondary"), SEEDS, strict=True):
            method = f"exp030_{role}_seed{seed}"
            seed_everything(seed)
            print(f"\n=== {EXPERIMENT_ID} {role.upper()} | seed={seed} ===", flush=True)
            train(
                data_dir=data_dir,
                fold=args.fold,
                splits_file=splits_path,
                method=method,
                n_epochs=args.epochs,
                lr=1e-4,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                unet_out_channels=32,
                unet_layers=[32, 64, 128],
                downsample=(1, 4, 4),
                det_loss_weight=1.0,
                det_neg_weight=0.02,
                max_iters=args.max_iters,
                seed=seed,
                window_size=3,
                pool_kernel_um=5.0,
                data_parallel=False,
            )
            checkpoint = starter_kit / "weights" / method / f"split_{args.fold}" / "edge_predictor_best.pth"
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
            write_manifest(running_path, receipt)

        receipt.update({"status": "COMPLETE", "completed_at": utc_now()})
        write_manifest(complete_path, receipt)
        print(f"\n{EXPERIMENT_ID} TRAINING COMPLETE: {complete_path}", flush=True)
        print("Do not push to Kaggle yet: run the EXP-030 ensemble validation first.", flush=True)
    except Exception as error:
        failure = {
            **receipt,
            "status": "FAILED",
            "failed_at": utc_now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        write_manifest(failed_path, failure)
        raise


if __name__ == "__main__":
    main()
