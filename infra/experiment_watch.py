#!/usr/bin/env python3
"""Token-free local watcher that periodically reconciles one remote run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from aienv import ensure_aienv

TERMINAL = {"DOC_PENDING", "CLOSED", "FAILED", "ORPHANED"}


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    controller = root / "infra" / "experimentctl.py"
    while True:
        result = subprocess.run(
            [sys.executable, str(controller), "reconcile", "--run-id", args.run_id],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip().splitlines()
        line = output[-1] if output else json.dumps({"ok": False, "error": result.stderr.strip()})
        print(line, flush=True)
        try:
            payload = json.loads(line)
            status = payload.get("runs", {}).get(args.run_id)
        except json.JSONDecodeError:
            status = None
        if status in TERMINAL:
            return
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()
