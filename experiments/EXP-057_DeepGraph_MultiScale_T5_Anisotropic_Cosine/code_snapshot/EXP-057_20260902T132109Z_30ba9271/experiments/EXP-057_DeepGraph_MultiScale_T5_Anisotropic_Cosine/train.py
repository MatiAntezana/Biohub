#!/usr/bin/env python3
"""EXP-057: DeepGraph MultiScale T5 Anisotropic Cosine (T=5 + Z 3.25um + WD 0.0003 + Cosine LR).

Both TemporalUNet3D + Transformer seeds train on every paired Biohub training
dataset with extended temporal context window T=5 and anisotropic Z spatial pooling (3.25um).
There is deliberately no validation holdout and therefore no local
validation score for this run. Execute only on a remote CUDA machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
)

EXPERIMENT_ID = "EXP-057"
PARENT_ID = "EXP-055"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "train"
DEFAULT_STARTER_KIT = PROJECT_ROOT / "official_starter_kit"
SEEDS = [3407, 9479]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {EXPERIMENT_ID}")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--starter-kit", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=750)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume-primary", type=Path, default=None)
    parser.add_argument("--resume-secondary", type=Path, default=None)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    experiment_dir = project_root / "experiments" / "EXP-057_DeepGraph_MultiScale_T5_Anisotropic_Cosine"
    data_dir = (args.data_dir or DEFAULT_DATA_DIR).resolve()
    starter_kit = (args.starter_kit or DEFAULT_STARTER_KIT).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    if not starter_kit.is_dir():
        raise FileNotFoundError(f"official starter kit not found: {starter_kit}")

    scripts_dir = starter_kit / "scripts"
    src_dir = starter_kit / "src"
    for candidate in (str(scripts_dir), str(src_dir), str(project_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    require_remote_cuda(
        experiment_id=EXPERIMENT_ID,
        allowed_nodes=["A1552", "A1555", "A1556", "A1557", "A1560"],
        force_remote=True,
    )

    from official_starter_kit.scripts.train_unet_transformer import train

    now_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = args.run_dir.name if args.run_dir is not None else f"{EXPERIMENT_ID}_{now_tag}_manual"
    artifact_dir = (args.run_dir or (experiment_dir / "artifacts" / run_id)).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    splits_path = artifact_dir / "full_fit_split.json"
    paired_datasets = write_full_fit_split(data_dir=data_dir, output_path=splits_path)

    plan = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_ID,
        "run_id": run_id,
        "project_root": str(project_root),
        "experiment_dir": str(experiment_dir),
        "artifact_dir": str(artifact_dir),
        "data_dir": str(data_dir),
        "starter_kit": str(starter_kit),
        "split_policy": "full-fit: 100% train / 0% validation",
        "validation": "none",
        "epochs_per_seed": args.epochs,
        "seeds": SEEDS,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_iters_per_epoch": args.max_iters,
        "checkpoint_selection": "last_epoch",
        "window_size": 5,
        "hidden_dim": 192,
        "n_heads": 6,
        "dropout": 0.1,
        "pool_kernel_um": 3.25,
        "unet_layers": [32, 64, 128],
        "unet_out_channels": 32,
        "downsample_zyx": [1, 2, 2],
        "det_loss_weight": 1.5,
        "det_negative_weight": 0.02,
        "ensemble": {
            "edge_secondary_weight": 0.15,
            "detection_secondary_weight": 0.475,
            "reverse_weight": 0.30,
            "retention_minimum": 0.90,
            "primary_margin_limit": 0.35,
            "std_ratio_minimum": 0.50,
            "std_ratio_maximum": 2.00,
        },
    }

    print(json.dumps(plan, indent=2))
    print(f"Paired datasets verified for full fit ({len(paired_datasets)}): {paired_datasets[:5]} ...", flush=True)

    running_path = artifact_dir / "RUNNING.json"
    complete_path = artifact_dir / "TRAINING_COMPLETE.json"
    failed_path = artifact_dir / "FAILED.json"
    receipt = {**plan, "started_at": utc_now(), "status": "RUNNING", "checkpoints": []}
    write_manifest(running_path, receipt)

    try:
        for role, seed in zip(("primary", "secondary"), SEEDS, strict=True):
            method = f"exp057_fullfit_{role}_seed{seed}"
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
                weight_decay=0.0003,
                lr_scheduler_type="cosine",
                min_lr=1e-5,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                unet_out_channels=32,
                unet_layers=[32, 64, 128],
                downsample=(1, 2, 2),
                det_loss_weight=1.5,
                det_neg_weight=0.02,
                max_iters=args.max_iters,
                seed=seed,
                window_size=5,
                hidden_dim=192,
                n_heads=6,
                dropout=0.1,
                pool_kernel_um=3.25,
                data_parallel=False,
                full_fit=True,
                output_dir=checkpoint_dir,
                progress=not args.compact,
                metrics_path=metrics_path,
                checkpoint_bundle=True,
                resume_from=resume_from,
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
            write_manifest(running_path, receipt)

        metric_summary = {}
        for role in ("primary", "secondary"):
            history = json.loads((artifact_dir / f"metrics_{role}.json").read_text(encoding="utf-8"))
            metric_summary[role] = history[-1] if history else {}
        receipt.update({"status": "COMPLETE", "completed_at": utc_now(), "metrics": metric_summary})
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
