#!/usr/bin/env python3
"""Export weights-only Kaggle artifacts from one canonical resumable checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from aienv import ensure_aienv


def main() -> None:
    ensure_aienv()
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("checkpoint no es un bundle resumible Biohub v1")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint no contiene model_state_dict")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"ok": True, "output": str(output), "parameters": len(state)}, sort_keys=True))


if __name__ == "__main__":
    main()
