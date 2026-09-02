#!/usr/bin/env python3
"""Compatibility entrypoint for the policy-gated Kaggle pipeline."""

from __future__ import annotations

import json

from aienv import ensure_aienv


def main() -> None:
    ensure_aienv()
    print(
        json.dumps(
            {
                "ok": False,
                "action": "kaggle-watchdog-deprecated",
                "error": "usar: python infra/kaggle_pipeline.py watch JOB_ID",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
