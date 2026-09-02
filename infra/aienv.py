#!/usr/bin/env python3
"""Enforce the canonical local Conda environment for Biohub operations."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ENVIRONMENT_NAME = "AIEnv"
REEXEC_MARKER = "BIOHUB_AIENV_REEXEC"


def in_aienv() -> bool:
    prefix = Path(sys.prefix).resolve()
    return os.environ.get("CONDA_DEFAULT_ENV") == ENVIRONMENT_NAME or prefix.name == ENVIRONMENT_NAME


def _conda_executable() -> Path | None:
    candidates = (
        Path.home() / "miniforge3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path("/opt/anaconda3/bin/conda"),
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def ensure_aienv() -> None:
    """Re-execute the current local command under ``conda run -n AIEnv``."""

    if in_aienv():
        return
    if os.environ.get(REEXEC_MARKER) == "1":
        raise RuntimeError("AIEnv no quedó activo después del reintento con Conda")
    conda = _conda_executable()
    if conda is None:
        raise RuntimeError("Conda no está disponible; se requiere `conda activate AIEnv`")
    os.environ[REEXEC_MARKER] = "1"
    arguments = [
        str(conda),
        "run",
        "--no-capture-output",
        "-n",
        ENVIRONMENT_NAME,
        "python",
        str(Path(sys.argv[0]).resolve()),
        *sys.argv[1:],
    ]
    os.execv(str(conda), arguments)

