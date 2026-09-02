#!/usr/bin/env python3
"""Rank comparable experiment evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TOOL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_ROOT))

from promotion_board import build_promotion_board, render_markdown  # noqa: E402


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=43200.0)
    args = parser.parse_args()

    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in args.evaluations]
    board = build_promotion_board(
        evaluations,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        min_coverage=args.min_coverage,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "promotion_board.json"
    markdown_path = args.output_dir / "promotion_board.md"
    json_path.write_text(
        json.dumps(json_safe(board), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(board), encoding="utf-8")
    print(render_markdown(board))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
