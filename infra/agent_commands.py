#!/usr/bin/env python3
"""Deterministic dispatcher for exact Spanish Biohub operator phrases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from aienv import ensure_aienv
from experimentlib import repo_root


COMMANDS = {
    "cortar train remoto": ["experimentctl.py", "stop", "--all"],
    "subir mejores versiones a kaggle": ["promote_best.py", "--limit", "3"],
    "modo automatico": ["automatic_mode.py", "supervise"],
    "cortar automatico": ["automatic_mode.py", "stop"],
}


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    args = parser.parse_args()
    key = normalize(args.instruction)
    if key not in COMMANDS:
        parser.error(f"orden no reconocida: {args.instruction}")
    root = repo_root()
    script, *arguments = COMMANDS[key]
    result = subprocess.run([sys.executable, str(root / "infra" / script), *arguments], cwd=root, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
