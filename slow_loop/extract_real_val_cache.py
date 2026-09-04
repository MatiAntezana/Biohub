#!/usr/bin/env python3
"""
Slow-Loop Validation Cache Extractor.
Extracts real candidate cell detections from the champion model (EXP-056)
and pairs them with ground truth lineages from .geff metadata to build
the Fast-Loop validation cache (fast_loop/val_cache.json).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td
import zarr
from geff import GeffMetadata
from tqdm import tqdm

# Ensure Biohub-mejorado root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import UNetNodeTransformer


def pool_kernel_from_um(um: float, voxel_size: Tuple[float, ...]) -> Tuple[int, ...]:
    """Convert physical suppression distance (microns) to per-axis voxel kernel."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def extract_peaks_3d(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.20,
    pool_kernel: Tuple[int, ...] = (3, 5, 5),
    downsample: Tuple[int, ...] = (1, 2, 2),
) -> List[Dict[str, Any]]:
    """
    Extracts local-max candidate detections from 3D logits.
    Returns coordinates scaled back to original image space.
    """
    logits = det_logits.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0]).cpu().numpy()  # (N, 3) in [z, y, x]

    if peak_idx.shape[0] == 0:
        return []

    probs = torch.sigmoid(det_logits).cpu().numpy()
    dz, dy, dx = downsample

    detections = []
    for i, (pz, py, px) in enumerate(peak_idx):
        conf = float(probs[pz, py, px])
        orig_pos = [float(pz * dz), float(py * dy), float(px * dx)]
        det_id = int(t * 1_000_000 + i)
        detections.append({
            "id": det_id,
            "t": int(t),
            "pos": orig_pos,
            "conf": round(conf, 5),
            "is_mitosis": 0.0,
        })
    return detections


def extract_validation_cache(
    zarr_path: Path,
    geff_path: Path,
    weights_path: Path,
    output_path: Path,
    max_frames: int | None = 100,
    downsample: Tuple[int, ...] = (1, 2, 2),
    pool_kernel_um: float = 3.25,
    det_threshold: float = 0.20,
    use_tta: bool = True,
    window_size: int = 4,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    print("=" * 80)
    print("🚀 EXTRACTING REAL VALIDATION CACHE FROM CHAMPION MODEL (EXP-056)")
    print("=" * 80)
    print(f"  Zarr Dataset : {zarr_path}")
    print(f"  Geff GT      : {geff_path}")
    print(f"  Weights      : {weights_path}")
    print(f"  Output Cache : {output_path}")
    print(f"  Max Frames   : {max_frames}")
    print(f"  Downsample   : {downsample}")
    print(f"  TTA Enabled  : {use_tta}")
    print(f"  Device       : {device_str}")
    print("-" * 80)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    t0_all = time.time()

    # 1. Open Zarr and inspect metadata
    z = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(z.attrs)
    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})
    q_low = float(quantiles.get("0.001", 70.0))
    q_high = float(quantiles.get("0.999", 3065.0))
    zarr_arr = z["0"]
    T_total, Z_orig, Y_orig, X_orig = zarr_arr.shape
    T = min(T_total, max_frames) if max_frames is not None else T_total
    print(f"  Full Zarr shape: {zarr_arr.shape}, processing T=0..{T-1}")
    print(f"  Quantiles: 0.001={q_low}, 0.999={q_high}")

    # Physical scale
    scale_orig = (1.625, 0.40625, 0.40625)
    ds_voxel_size = tuple(s * d for s, d in zip(scale_orig, downsample))
    pool_k = pool_kernel_from_um(pool_kernel_um, ds_voxel_size)
    print(f"  Suppression pool kernel for {pool_kernel_um} µm: {pool_k}")

    # 2. Load Ground Truth from GEFF
    print(f"  Loading ground truth from {geff_path}...")
    geff_res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    gt_graph = geff_res[0] if isinstance(geff_res, tuple) else geff_res
    meta = GeffMetadata.read(str(geff_path))
    estimated_total_embryo_nodes = int((meta.extra or {}).get("estimated_number_of_nodes", 58672))
    est_nodes_per_frame_val = int(round(estimated_total_embryo_nodes / T_total))

    gt_df = gt_graph.node_attrs()
    gt_nodes_by_t: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(T)}
    valid_gt_node_ids = set()

    for row in gt_df.iter_rows(named=True):
        t_frame = int(row["t"])
        if t_frame < T:
            nid = int(row["node_id"])
            valid_gt_node_ids.add(nid)
            gt_nodes_by_t[t_frame].append({
                "id": nid,
                "t": t_frame,
                "pos": [float(row["z"]), float(row["y"]), float(row["x"])],
            })

    gt_edges_raw = gt_graph.edge_list()
    gt_edges: List[List[int]] = []
    for src, tgt in gt_edges_raw:
        s_id, t_id = int(src), int(tgt)
        if s_id in valid_gt_node_ids and t_id in valid_gt_node_ids:
            gt_edges.append([s_id, t_id])

    estimated_nodes_by_t: Dict[int, int] = {t: est_nodes_per_frame_val for t in range(T)}
    print(f"  Loaded GT: {sum(len(v) for v in gt_nodes_by_t.values())} nodes, {len(gt_edges)} edges across {T} frames.")
    print(f"  Estimated embryo cells per frame: {est_nodes_per_frame_val} (Total {estimated_total_embryo_nodes} for embryo)")

    # 3. Load EXP-056 Model
    print("  Instantiating and loading UNetNodeTransformer...")
    model = UNetNodeTransformer(
        unet_out_channels=48,
        unet_layers=(48, 96, 192),
        hidden_dim=192,
        n_heads=6,
        n_blocks=4,
    )
    sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print("  Weights loaded strictly into GPU memory!")

    # 4. Run Sliding Window Feature & Peak Extraction
    W = window_size
    stride = W - 1
    window_starts = list(range(0, max(1, T - W + 1), stride))
    if window_starts[-1] + W < T:
        window_starts.append(T - W)

    dz, dy, dx = downsample
    raw_detections_by_t: Dict[int, List[Dict[str, Any]]] = {}
    processed_frames: set[int] = set()

    print(f"  Executing forward passes over {len(window_starts)} temporal windows...")
    t_inf_start = time.time()

    with torch.no_grad():
        for ws in tqdm(window_starts, desc="Inference Windows"):
            frame_indices = [t for t in range(ws, ws + W) if t < T]
            if len(frame_indices) < W:
                frame_indices = list(range(max(0, T - W), T))

            # Load downsampled frames
            frames_np = []
            for t_idx in frame_indices:
                raw_frame = zarr_arr[t_idx, ::dz, ::dy, ::dx].astype(np.float32)
                frames_np.append(raw_frame)
            
            imgs = torch.from_numpy(np.stack(frames_np))  # (W, Z, Y, X)
            # Quantile normalisation
            imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
            imgs = imgs.unsqueeze(0).to(device)  # (1, W, Z, Y, X)

            unet_out, det_logits = model.encode(imgs)

            if use_tta:
                _nv = 1
                for dims in [(-1,), (-2,), (-2, -1)]:
                    imgs_flip = imgs.flip(dims)
                    _, det_flip = model.encode(imgs_flip)
                    for f in range(W):
                        det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                    _nv += 1
                for _k in (1, 3):
                    imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                    _, det_rot = model.encode(imgs_rot)
                    for f in range(W):
                        det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                    _nv += 1
                imgs_t = imgs.transpose(-1, -2)
                _, det_t = model.encode(imgs_t)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
                _nv += 1
                imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
                _, det_at = model.encode(imgs_at)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
                _nv += 1
                for f in range(W):
                    det_logits[f] = det_logits[f] / _nv

            for f_idx, t_idx in enumerate(frame_indices):
                if t_idx not in processed_frames and t_idx < T:
                    peaks = extract_peaks_3d(
                        det_logits=det_logits[f_idx][0, 0],
                        t=t_idx,
                        det_threshold=det_threshold,
                        pool_kernel=pool_k,
                        downsample=downsample,
                    )
                    raw_detections_by_t[t_idx] = peaks
                    processed_frames.add(t_idx)

    t_inf = time.time() - t_inf_start
    total_candidates = sum(len(d) for d in raw_detections_by_t.values())
    avg_per_frame = total_candidates / max(1, len(raw_detections_by_t))
    print(f"  Inference done in {t_inf:.2f}s ({t_inf/T:.3f}s/frame)")
    print(f"  Extracted {total_candidates} total peak candidates (avg {avg_per_frame:.1f}/frame)")

    # 5. Compile and Save Cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "raw_detections": raw_detections_by_t,
        "gt_nodes": gt_nodes_by_t,
        "gt_edges": gt_edges,
        "estimated_nodes_per_frame": estimated_nodes_by_t,
        "metadata": {
            "dataset_name": zarr_path.stem,
            "model_weights": weights_path.name,
            "n_timepoints": T,
            "total_raw_detections": total_candidates,
            "total_gt_nodes": sum(len(n) for n in gt_nodes_by_t.values()),
            "total_gt_edges": len(gt_edges),
            "estimated_total_nodes": sum(estimated_nodes_by_t.values()),
            "det_threshold": det_threshold,
            "pool_kernel_um": pool_kernel_um,
            "downsample": list(downsample),
            "use_tta": use_tta,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    print(f"  Saving validation cache to: {output_path}...")
    with open(output_path, "w") as f:
        json.dump(cache_payload, f)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Successfully wrote {file_size_mb:.2f} MB validation cache!")
    print(f"  Total pipeline wall time: {time.time() - t0_all:.2f}s")
    print("=" * 80)
    return cache_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Real Validation Cache from EXP-056 Champion Model")
    parser.add_argument("--zarr-path", type=Path, default=Path("/home/alumno1/Documents/Biohub/data/train/44b6_12dfb391.zarr"))
    parser.add_argument("--geff-path", type=Path, default=Path("/home/alumno1/Documents/Biohub/data/train/44b6_12dfb391.geff"))
    parser.add_argument("--weights-path", type=Path, default=Path("/home/alumno1/Documents/Biohub-mejorado/champion_weights/experiments/EXP-056_DeepGraph_Anisotropic_Wide_Cosine/artifacts/deployment/weights_dataset/exp056_primary_weights.pt"))
    parser.add_argument("--output-cache", type=Path, default=PROJECT_ROOT / "fast_loop" / "val_cache.json")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--det-threshold", type=float, default=0.20)
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA for faster extraction")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    extract_validation_cache(
        zarr_path=args.zarr_path,
        geff_path=args.geff_path,
        weights_path=args.weights_path,
        output_path=args.output_cache,
        max_frames=args.max_frames,
        det_threshold=args.det_threshold,
        use_tta=not args.no_tta,
        device_str=args.device,
    )
