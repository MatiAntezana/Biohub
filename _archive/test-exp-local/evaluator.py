#!/usr/bin/env python3
"""Run real Biohub inference and the official metric for legacy heatmap models.

This is the canonical local/remote evaluator entry point. It performs actual
checkpoint inference and delegates scoring to the official Biohub metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--architecture",
        choices=(
            "anisotropic_unet3d",
            "wide_unet3d",
            "wide_resunet3d",
            "stardist3d",
        ),
        required=True,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Defaults to 64 for wide architectures and 32 otherwise.",
    )
    parser.add_argument(
        "--tta",
        choices=("identity", "flip_x"),
        default="flip_x",
        help="flip_x averages identity and an X-axis flip; identity is single-pass.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-link-dist", type=float, default=12.0)
    parser.add_argument("--max-gap-dist", type=float, default=16.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--max-match-distance", type=float, default=7.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    official_root = project_root / "official_starter_kit"

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(official_root / "src"))
    sys.path.insert(0, str(official_root / "scripts"))

    import numpy as np
    import torch
    import zarr
    from geff import GeffMetadata
    from skimage.feature import peak_local_max

    import tracksdata as td
    from csv_to_geffs import csv_to_geffs
    from evaluate import _load_graph, _read_scale
    from tracking_cellmot.metrics import (
        evaluate as compute_metric,
        node_recall,
        per_sample_metrics,
        summarise,
    )
    from src.tracking.gap_closing_tracker import track_with_gap_closing

    default_channels = 64 if args.architecture.startswith("wide_") else 32
    base_channels = args.base_channels or default_channels
    if args.architecture in {"anisotropic_unet3d", "wide_unet3d"}:
        from src.models.unet3d_anisotropic import AnisotropicUNet3D

        model = AnisotropicUNet3D(
            in_channels=1, out_channels=1, base_channels=base_channels
        )
    elif args.architecture == "wide_resunet3d":
        from src.models.resnet_unet3d import ResUNet3D

        model = ResUNet3D(in_channels=1, out_channels=1, base_channels=base_channels)
    else:
        from src.models.stardist3d import StarDist3D

        model = StarDist3D(in_channels=1, n_rays=32, base_channels=base_channels)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3D evaluation")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    samples = list(protocol["samples"])
    missing = [
        name
        for name in samples
        if not (data_dir / f"{name}.zarr").exists()
        or not (data_dir / f"{name}.geff").exists()
    ]
    if missing:
        raise FileNotFoundError(f"protocol datasets missing image/GT pairs: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    geff_dir = output_dir / "predicted_geffs"
    prediction_csv = output_dir / "predictions.csv"
    metrics_path = output_dir / "evaluation.json"
    device = torch.device("cuda")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
    fieldnames = [
        "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
    ]
    all_rows: list[dict[str, Any]] = []
    sample_runtimes: dict[str, float] = {}
    started = time.perf_counter()
    print(
        f"Evaluating {args.experiment_id}: {len(samples)} frozen holdout datasets, "
        f"architecture={args.architecture}, threshold={args.threshold}",
        flush=True,
    )

    with torch.inference_mode():
        for sample_index, name in enumerate(samples, start=1):
            sample_started = time.perf_counter()
            image = zarr.open(data_dir / f"{name}.zarr", mode="r")["0"]
            nodes_by_t: dict[int, list[tuple[int, tuple[int, int, int]]]] = {}
            node_id = 1
            for frame in range(image.shape[0]):
                volume = np.asarray(image[frame], dtype=np.float32)
                p1, p998 = np.percentile(volume, (1.0, 99.8))
                volume = np.clip(
                    (volume - p1) / max(1e-5, p998 - p1), 0.0, 1.0
                ).astype(np.float32, copy=False)
                tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
                def heatmap_output(model_output: Any) -> Any:
                    if isinstance(model_output, dict):
                        if "prob" in model_output:
                            model_output = model_output["prob"]
                        elif "main" in model_output:
                            model_output = model_output["main"]
                        else:
                            raise KeyError(
                                f"unsupported model output keys: {sorted(model_output)}"
                            )
                    return model_output.squeeze(0).squeeze(0)

                with torch.amp.autocast("cuda"):
                    original = heatmap_output(model(tensor))
                    if args.tta == "flip_x":
                        flipped_x = torch.flip(
                            heatmap_output(model(torch.flip(tensor, dims=(-1,)))),
                            dims=(-1,),
                        )
                        predicted = (original + flipped_x) / 2.0
                    else:
                        predicted = original
                    heatmap = predicted.float().cpu().numpy()
                peaks = peak_local_max(
                    heatmap,
                    min_distance=2,
                    threshold_abs=args.threshold,
                    exclude_border=False,
                )
                nodes_by_t[frame] = []
                for z_coord, y_coord, x_coord in peaks:
                    nodes_by_t[frame].append(
                        (node_id, (int(z_coord), int(y_coord), int(x_coord)))
                    )
                    node_id += 1

            filtered_nodes, edges = track_with_gap_closing(
                nodes_by_t,
                scale=scale,
                max_link_dist=args.max_link_dist,
                enable_gap_closing=True,
                max_gap_dist=args.max_gap_dist,
                min_track_length=args.min_track_length,
                enable_mitosis=False,
            )
            for predicted_id, frame, z_coord, y_coord, x_coord in filtered_nodes:
                all_rows.append(
                    {
                        "dataset": name,
                        "row_type": "node",
                        "node_id": predicted_id,
                        "t": frame,
                        "z": z_coord,
                        "y": y_coord,
                        "x": x_coord,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
            for source_id, target_id in edges:
                all_rows.append(
                    {
                        "dataset": name,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )
            torch.cuda.synchronize()
            sample_runtimes[name] = time.perf_counter() - sample_started
            print(
                f"[{sample_index:02d}/{len(samples):02d}] {name}: "
                f"nodes={len(filtered_nodes)} edges={len(edges)} "
                f"time={sample_runtimes[name]:.1f}s",
                flush=True,
            )

    with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    csv_to_geffs(prediction_csv, geff_dir, overwrite=True)

    per_sample: list[dict[str, Any]] = []
    skipped: list[str] = []
    for name in samples:
        try:
            predicted_graph = _load_graph(geff_dir / f"{name}.geff")
            ground_truth = _load_graph(data_dir / f"{name}.geff")
            voxel_scale = _read_scale(data_dir, name)
            result = compute_metric(
                predicted_graph,
                ground_truth,
                scale=voxel_scale,
                max_distance=args.max_match_distance,
            )
            recall = node_recall(predicted_graph, ground_truth)
            metadata = GeffMetadata.read(data_dir / f"{name}.geff")
            estimated_nodes = (metadata.extra or {}).get("estimated_number_of_nodes")
            estimated_nodes = float(estimated_nodes) if estimated_nodes is not None else float("nan")
            row = per_sample_metrics(result, estimated_nodes, recall)
            row.update(
                {
                    "dataset": name,
                    "estimated_number_of_nodes": estimated_nodes,
                    "inference_seconds": sample_runtimes[name],
                }
            )
            per_sample.append(row)
            print(
                f"METRIC {name}: adj_edge={row['adj_edge_jaccard']:.4f} "
                f"edge={row['edge_jaccard']:.4f} recall={row['node_recall']:.4f}",
                flush=True,
            )
        except Exception as error:
            skipped.append(name)
            print(f"SKIP {name}: {type(error).__name__}: {error}", flush=True)

    total_seconds = time.perf_counter() - started
    summary = summarise(per_sample)
    payload = {
        "schema_version": 1,
        "status": "SUCCESS" if not skipped and len(per_sample) == len(samples) else "PARTIAL",
        "experiment_id": args.experiment_id,
        "protocol_id": protocol["protocol_id"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "requested_dataset_count": len(samples),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "inference_config": {
            "architecture": args.architecture,
            "base_channels": base_channels,
            "threshold": args.threshold,
            "tta": ["identity", "flip_x"] if args.tta == "flip_x" else ["identity"],
            "max_link_dist_um": args.max_link_dist,
            "max_gap_dist_um": args.max_gap_dist,
            "min_track_length": args.min_track_length,
            "mitosis": False,
            "max_match_distance_um": args.max_match_distance,
        },
        "runtime": {
            "total_seconds": total_seconds,
            "per_dataset_seconds": sample_runtimes,
        },
        "summary": summary,
        "per_sample": per_sample,
        "skipped": skipped,
        "protocol_limitations": protocol.get("limitations", []),
    }
    metrics_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n=== OFFICIAL SUMMARY ===", flush=True)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)
    print(f"RESULT_JSON={metrics_path}", flush=True)
    if payload["status"] != "SUCCESS":
        raise RuntimeError(f"evaluation incomplete: {skipped}")


if __name__ == "__main__":
    main()
