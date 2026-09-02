"""Validate a real Biohub evaluation artifact.

The previous implementation estimated scores from log text and fixed
constants. That behavior was invalid and has been removed. Checkpoints must
first be evaluated with ``test-exp-local/evaluator.py`` (or
an architecture-specific equivalent) against labeled GEFF ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_ROOT))

from promotion_board import summarise_rows


def validate_official_evaluation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCESS":
        raise ValueError(f"evaluation is not complete: status={payload.get('status')!r}")
    rows = payload.get("per_sample")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation contains no per-sample official metric rows")
    required = {
        "dataset",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "node_recall",
        "adj_edge_jaccard",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"row {row.get('dataset')} missing fields: {sorted(missing)}")
    recomputed = summarise_rows(rows)
    if not math.isfinite(float(recomputed["score"])):
        raise ValueError("official score is not finite")
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_official_evaluation(args.evaluation_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
