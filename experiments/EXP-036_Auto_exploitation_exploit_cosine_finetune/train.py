#!/usr/bin/env python3
"""Generated entrypoint for a policy-controlled Biohub variant."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.autonomous_detection_training import run


if __name__ == "__main__":
    run([*sys.argv[1:], "--variant-file", str(Path(__file__).with_name("variant.json"))])
