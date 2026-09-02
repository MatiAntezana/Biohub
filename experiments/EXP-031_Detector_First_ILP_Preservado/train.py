#!/usr/bin/env python3
"""EXP-031: detector-first TemporalUNet+Transformer with hard negatives.

This audit-driven branch trains one model from scratch on every paired
training dataset, with no validation holdout. Its checkpoint is intended for
ILP-preserving inference: solver edges
remain authoritative and motion/division rules may only add high-confidence
links among unmatched nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from functools import partial
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


EXPERIMENT_ID = "EXP-031"
SEED = 7319


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--starter-kit", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=850)
    parser.add_argument("--hard-negative-ratio", type=int, default=48)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
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
        "project_root": str(project_root),
        "data_dir": str(data_dir),
        "starter_kit": str(starter_kit),
        "artifact_dir": str(artifact_dir),
        "experiment_dir": str(experiment_dir),
        "run_id": artifact_dir.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_iters": args.max_iters,
        "seed": SEED,
        "split_policy": "full-fit: 100% train / 0% validation",
        "validation": "none",
        "checkpoint_selection": "last_epoch",
        "window_size": 3,
        "downsample_zyx": [1, 4, 4],
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "det_loss_weight": 2.0,
        "det_negative_weight": 0.25,
        "hard_negative_ratio": args.hard_negative_ratio,
        "hard_negative_ignore_kernel_zyx": [3, 5, 5],
        "repair_policy": {
            "scale_um": [1.625, 0.40625, 0.40625],
            "min_ilp_confidence": 0.48,
            "tight_motion_gate_um": 6.0,
            "relaxed_motion_gate_um": 10.0,
            "velocity_weight": 0.5,
            "learned_probability_bonus": 1.0,
            "raw_distance_weight": 0.05,
            "division_parent_candidate_um": 8.0,
            "division_existing_child_um": 10.0,
            "division_sibling_um": 11.0,
            "division_min_angle_deg": 35.0,
            "division_min_separation_growth_um": 2.25,
        },
    }


def main() -> None:
    args = parse_args()
    plan = resolved_plan(args)
    print(json.dumps(plan, sort_keys=True) if args.compact else json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "DRY_RUN_OK"}), flush=True)
        return
    if args.run_dir is None:
        raise ValueError("--run-dir es obligatorio fuera de --dry-run; usar experimentctl run")

    require_remote_cuda()
    data_dir = Path(plan["data_dir"])
    starter_kit = Path(plan["starter_kit"])
    artifact_dir = Path(plan["artifact_dir"])
    experiment_dir = Path(plan["experiment_dir"])
    try:
        artifact_dir.relative_to(experiment_dir / "artifacts")
    except ValueError as error:
        raise ValueError("run-dir debe vivir dentro de artifacts/ del experimento") from error
    scripts_dir = starter_kit / "scripts"
    source_dir = starter_kit / "src"
    if not scripts_dir.is_dir() or not source_dir.is_dir():
        raise FileNotFoundError(f"Incomplete starter kit under {starter_kit}")

    splits_path = artifact_dir / "full_fit_split.json"
    full_fit_split = write_full_fit_split(data_dir, splits_path)[0]
    plan["train_dataset_count"] = len(full_fit_split["train"])

    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(scripts_dir))
    from train_unet_transformer import train  # noqa: PLC0415
    from src.strategies.detector_precision import precision_focal_detection_loss

    method = f"exp031_detector_first_fullfit_seed{SEED}"
    checkpoint_dir = experiment_dir / "checkpoints" / plan["run_id"] / "model"
    metrics_path = artifact_dir / "metrics.json"
    running_path = artifact_dir / "RUNNING.json"
    complete_path = artifact_dir / "TRAINING_COMPLETE.json"
    failed_path = artifact_dir / "FAILED.json"
    receipt = {**plan, "started_at": utc_now(), "status": "RUNNING"}
    write_manifest(running_path, receipt)

    loss_hook = partial(
        precision_focal_detection_loss,
        hard_negative_ratio=args.hard_negative_ratio,
        gamma=2.0,
        ignore_kernel=(3, 5, 5),
    )
    try:
        seed_everything(SEED)
        train(
            data_dir=data_dir,
            fold=0,
            splits_file=splits_path,
            method=method,
            n_epochs=args.epochs,
            lr=1e-4,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            unet_out_channels=32,
            unet_layers=[32, 64, 128],
            downsample=(1, 4, 4),
            det_loss_weight=2.0,
            det_neg_weight=0.25,
            max_iters=args.max_iters,
            seed=SEED,
            window_size=3,
            pool_kernel_um=4.0,
            data_parallel=False,
            detection_loss_fn=loss_hook,
            full_fit=True,
            output_dir=checkpoint_dir,
            progress=not args.compact,
            metrics_path=metrics_path,
            checkpoint_bundle=True,
            resume_from=args.resume,
        )
        checkpoint = checkpoint_dir / "training_checkpoint.pth"
        config_path = checkpoint.parent / "config.json"
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"training finished without expected checkpoint: {checkpoint}")
        history = json.loads(metrics_path.read_text(encoding="utf-8"))
        receipt.update(
            {
                "status": "COMPLETE",
                "completed_at": utc_now(),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_config": str(config_path),
                "metrics": history[-1] if history else {},
            }
        )
        write_manifest(complete_path, receipt)
        print(f"\n{EXPERIMENT_ID} TRAINING COMPLETE: {complete_path}", flush=True)
        print("No local validation score exists for this full-fit run; evaluate via the downstream test/Kaggle gate.", flush=True)
    except Exception as error:
        write_manifest(
            failed_path,
            {
                **receipt,
                "status": "FAILED",
                "failed_at": utc_now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
