"""Runtime helpers shared by the remote EXP-030 and EXP-031 launchers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch before constructing a model or loader."""

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_remote_cuda():
    """Refuse accidental training on the user's local CPU-only machine."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. This experiment must run on a remote GPU node; "
            "do not execute training on the local Mac."
        )
    return torch.device("cuda")


def dataset_stems(data_dir: Path) -> list[str]:
    """Return datasets for which both image and lineage annotation exist."""

    stems = []
    for zarr_path in sorted(data_dir.glob("*.zarr")):
        stem = zarr_path.name.removesuffix(".zarr")
        if (data_dir / f"{stem}.geff").exists():
            stems.append(stem)
    if not stems:
        raise FileNotFoundError(
            f"No paired .zarr/.geff training datasets found under {data_dir}"
        )
    return stems


def embryo_group(stem: str) -> str:
    """Extract the embryo identifier used for group-disjoint validation."""

    return stem.split("_", 1)[0]


def write_embryo_disjoint_splits(data_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    """Create deterministic leave-one-embryo-out folds.

    The competition data currently exposes embryo prefixes such as ``6bba``
    and ``44b6``.  Keeping a prefix entirely in one side prevents temporal or
    biological leakage between training and validation clips.
    """

    stems = dataset_stems(data_dir)
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(embryo_group(stem), []).append(stem)
    if len(groups) < 2:
        raise ValueError(
            "Embryo-disjoint validation needs at least two dataset prefixes; "
            f"found {sorted(groups)}"
        )

    folds: list[dict[str, Any]] = []
    all_stems = set(stems)
    for fold_index, held_out_group in enumerate(sorted(groups)):
        val = sorted(groups[held_out_group])
        train = sorted(all_stems.difference(val))
        folds.append(
            {
                "split": fold_index,
                "held_out_embryo": held_out_group,
                "train": train,
                "test": val,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    return folds


def sha256_file(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON receipt for a completed remote stage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
