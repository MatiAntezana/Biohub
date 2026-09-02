#!/usr/bin/env python3
"""Rank eligible experiments and push up to three best versions to Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import experiment_directories, latest_events_by_run, load_manifest, read_json, repo_root
from kaggle_pipeline import JobStore, KagglePipeline, static_preflight


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def ranked_candidates(root: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = root or repo_root()
    policy = read_json(root / ".agents" / "ranking.json", {})
    precedence = policy.get("metric_precedence", [])
    measurements: dict[str, list[tuple[str, float, str]]] = {}
    board_path = root / "experiments" / "Test_Exp_Local_EXP015_EXP017_EXP019_EXP022" / "board" / "promotion_board.json"
    board = read_json(board_path, {})
    for candidate in board.get("candidates", []):
        score = _number(candidate.get("recomputed_summary", {}).get("score"))
        if score is not None and candidate.get("eligible") is True:
            measurements.setdefault(candidate["experiment_id"], []).append(
                ("global_score", score, board.get("generated_at", "2026-08-28T00:00:00Z"))
            )
    jobs = JobStore(root).read().get("jobs", {})
    for job in jobs.values():
        score = _number(job.get("public_score"))
        if score is not None:
            measurements.setdefault(job["experiment_id"], []).append(("public_score", score, job.get("updated_at", "")))
    for event in latest_events_by_run(root).values():
        metrics = event.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric in precedence:
            score = _number(metrics.get(metric))
            if score is not None:
                measurements.setdefault(event["experiment_id"], []).append((metric, score, event.get("timestamp_utc", "")))

    required_cohort = int(policy.get("promotion_limit", 3))
    counts = {
        metric: sum(any(row[0] == metric for row in values) for values in measurements.values())
        for metric in precedence
    }
    cohort_metric = next((metric for metric in precedence if counts.get(metric, 0) >= required_cohort), None)
    if cohort_metric is None and precedence:
        cohort_metric = sorted(precedence, key=lambda metric: (-counts.get(metric, 0), precedence.index(metric)))[0]

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for experiment_id in sorted(experiment_directories(root)):
        try:
            _, manifest = load_manifest(experiment_id, root)
            kaggle = manifest.get("kaggle", {})
        except Exception:
            kaggle = read_json(experiment_directories(root)[experiment_id] / "kaggle" / "deployment.json", {})
            if not kaggle:
                excluded.append({"experiment_id": experiment_id, "reason": "sin manifiesto de deployment"})
                continue
        if kaggle.get("deployment_ready") is not True:
            excluded.append({"experiment_id": experiment_id, "reason": "deployment_ready=false"})
            continue
        errors = static_preflight(experiment_id, root)
        if errors:
            excluded.append({"experiment_id": experiment_id, "reason": "; ".join(errors)})
            continue
        values = [row for row in measurements.get(experiment_id, []) if row[0] == cohort_metric]
        if not values:
            excluded.append({"experiment_id": experiment_id, "reason": f"sin métrica comparable {cohort_metric}"})
            continue
        metric, score, measured_at = sorted(values, key=lambda row: (-row[1], row[2]))[0]
        candidates.append(
            {
                "experiment_id": experiment_id,
                "metric": metric,
                "score": score,
                "measured_at": measured_at,
            }
        )
    reverse = policy.get("higher_is_better", True) is True
    candidates.sort(key=lambda row: (row["score"], row["measured_at"]), reverse=reverse)
    return candidates, excluded


def promote(limit: int = 3, *, dry_run: bool = False, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    candidates, excluded = ranked_candidates(root)
    selected = candidates[:limit]
    pipeline = KagglePipeline(root)
    jobs = [pipeline.push(row["experiment_id"], dry_run=dry_run) for row in selected]
    return {"selected": selected, "jobs": jobs, "excluded": excluded, "dry_run": dry_run}


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = promote(max(1, min(args.limit, 3)), dry_run=args.dry_run)
    print(json.dumps({"ok": True, "action": "promote-best", **result}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
