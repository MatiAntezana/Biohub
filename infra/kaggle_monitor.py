#!/usr/bin/env python3
"""Low-noise monitor for one or more Kaggle kernels.

The monitor is deliberately read-only: it records status and output metadata,
but never pushes a kernel or submits to a competition.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import atomic_write_json, utc_now


TERMINAL = ("COMPLETE", "ERROR", "FAILED", "CANCEL")


def kernel_status(kernel: str) -> dict[str, Any]:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or result.stderr).strip()
    upper = text.upper()
    state = next((name for name in TERMINAL if name in upper), "RUNNING" if "RUNNING" in upper else "UNKNOWN")
    return {
        "kernel": kernel,
        "state": state,
        "checked_at": utc_now(),
        "returncode": result.returncode,
        "message": text[-1000:],
    }


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", action="append", required=True, help="Kaggle slug owner/kernel; repeatable")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--state-file", type=Path, default=Path(".state/kaggle-monitor.json"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    state_file = args.state_file.expanduser().resolve()
    last_states: dict[str, str] = {}
    while True:
        results = [kernel_status(kernel) for kernel in args.kernel]
        atomic_write_json(state_file, {"schema_version": 1, "updated_at": utc_now(), "kernels": results})
        for result in results:
            if last_states.get(result["kernel"]) != result["state"]:
                print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
            last_states[result["kernel"]] = result["state"]
        if args.once or all(result["state"] in TERMINAL for result in results):
            return
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()
