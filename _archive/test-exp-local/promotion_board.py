"""Build a conservative Kaggle promotion board from real evaluation rows.

The module intentionally has no ML dependencies. It consumes per-dataset
TP/FP/FN counts produced by the official Biohub metric implementation and
recomputes every aggregate instead of trusting a copied headline score.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any, Iterable


ADJUSTMENT_ALPHA = 0.1
DIVISION_WEIGHT = 0.1


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator > 0 else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the official run-level aggregate from per-dataset rows."""

    valid = [row for row in rows if _finite(row.get("edge_tp"))]
    if not valid:
        return {
            "n": 0,
            "edge_jaccard": float("nan"),
            "adj_edge_jaccard": float("nan"),
            "division_jaccard": float("nan"),
            "node_recall": float("nan"),
            "score": float("nan"),
        }

    edge_tp = sum(int(row["edge_tp"]) for row in valid)
    edge_fp = sum(int(row["edge_fp"]) for row in valid)
    edge_fn = sum(int(row["edge_fn"]) for row in valid)
    division_tp = sum(int(row["division_tp"]) for row in valid)
    division_fp = sum(int(row["division_fp"]) for row in valid)
    division_fn = sum(int(row["division_fn"]) for row in valid)

    adjusted_weighted_sum = 0.0
    adjusted_weight = 0
    for row in valid:
        weight = int(row["edge_tp"]) + int(row["edge_fp"]) + int(row["edge_fn"])
        adjusted = row.get("adj_edge_jaccard")
        if weight > 0 and _finite(adjusted):
            adjusted_weighted_sum += weight * float(adjusted)
            adjusted_weight += weight

    adj_edge = (
        adjusted_weighted_sum / adjusted_weight
        if adjusted_weight > 0
        else float("nan")
    )
    division_total = division_tp + division_fp + division_fn
    division_jaccard = (
        _jaccard(division_tp, division_fp, division_fn)
        if division_total > 0
        else float("nan")
    )
    score = (
        adj_edge + DIVISION_WEIGHT * division_jaccard
        if _finite(adj_edge) and _finite(division_jaccard)
        else adj_edge
    )
    recalls = [float(row["node_recall"]) for row in valid if _finite(row.get("node_recall"))]
    return {
        "n": len(valid),
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "division_tp": division_tp,
        "division_fp": division_fp,
        "division_fn": division_fn,
        "edge_jaccard": _jaccard(edge_tp, edge_fp, edge_fn),
        "adj_edge_jaccard": adj_edge,
        "division_jaccard": division_jaccard,
        "node_recall": sum(recalls) / len(recalls) if recalls else float("nan"),
        "score": score,
    }


def _bootstrap_summary(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(iterations):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        score = summarise_rows(sampled)["score"]
        if _finite(score):
            scores.append(float(score))
    return {
        "iterations": len(scores),
        "score_ci_low": _percentile(scores, 0.025),
        "score_ci_high": _percentile(scores, 0.975),
    }


def _sample_score(row: dict[str, Any]) -> float:
    adjusted = row.get("adj_edge_jaccard")
    if not _finite(adjusted):
        return float("nan")
    division_total = (
        int(row["division_tp"]) + int(row["division_fp"]) + int(row["division_fn"])
    )
    if division_total == 0:
        return float(adjusted)
    division = _jaccard(
        int(row["division_tp"]), int(row["division_fp"]), int(row["division_fn"])
    )
    return float(adjusted) + DIVISION_WEIGHT * division


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row["dataset"]).split("_", 1)[0]
        grouped.setdefault(group, []).append(row)
    return {group: summarise_rows(group_rows) for group, group_rows in sorted(grouped.items())}


def _paired_delta(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_rows = {row["dataset"]: row for row in left["per_sample"]}
    right_rows = {row["dataset"]: row for row in right["per_sample"]}
    names = sorted(set(left_rows) & set(right_rows))
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled_names = [names[rng.randrange(len(names))] for _ in names]
        left_score = summarise_rows([left_rows[name] for name in sampled_names])["score"]
        right_score = summarise_rows([right_rows[name] for name in sampled_names])["score"]
        if _finite(left_score) and _finite(right_score):
            deltas.append(float(left_score) - float(right_score))
    point_delta = (
        float(left["recomputed_summary"]["score"])
        - float(right["recomputed_summary"]["score"])
    )
    return {
        "left": left["experiment_id"],
        "right": right["experiment_id"],
        "n_paired": len(names),
        "score_delta": point_delta,
        "delta_ci_low": _percentile(deltas, 0.025),
        "delta_ci_high": _percentile(deltas, 0.975),
        "probability_left_better": (
            sum(delta > 0 for delta in deltas) / len(deltas) if deltas else float("nan")
        ),
    }


def build_promotion_board(
    evaluations: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 20260827,
    min_coverage: float = 1.0,
    max_runtime_seconds: float = 12 * 60 * 60,
) -> dict[str, Any]:
    """Rank experiments using the conservative lower confidence bound.

    All candidates must use the same protocol and exactly the same dataset
    names. Operational diagnostics act only as hard gates; they are never
    mixed into an invented weighted quality score.
    """

    if len(evaluations) < 1:
        raise ValueError("at least one evaluation is required")
    protocols = {evaluation.get("protocol_id") for evaluation in evaluations}
    if len(protocols) != 1:
        raise ValueError(f"evaluations use different protocols: {sorted(protocols)}")
    sample_sets = [
        {row["dataset"] for row in evaluation.get("per_sample", [])}
        for evaluation in evaluations
    ]
    if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
        raise ValueError("evaluations do not contain exactly the same dataset names")

    candidates: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for index, evaluation in enumerate(evaluations):
        recomputed = summarise_rows(evaluation.get("per_sample", []))
        requested = int(evaluation.get("requested_dataset_count") or len(sample_sets[index]))
        coverage = recomputed["n"] / requested if requested else 0.0
        runtime = float(evaluation.get("runtime", {}).get("total_seconds", float("inf")))
        failed_gates = []
        if evaluation.get("status") != "SUCCESS":
            failed_gates.append("evaluation_status")
        if coverage < min_coverage:
            failed_gates.append("coverage")
        if not _finite(recomputed["score"]):
            failed_gates.append("finite_official_score")
        if runtime > max_runtime_seconds:
            failed_gates.append("runtime")
        if evaluation.get("skipped"):
            failed_gates.append("skipped_datasets")

        bootstrap = _bootstrap_summary(
            evaluation["per_sample"],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + index,
        )
        sample_scores = [
            (row["dataset"], _sample_score(row)) for row in evaluation["per_sample"]
        ]
        finite_sample_scores = [item for item in sample_scores if _finite(item[1])]
        worst_dataset, worst_dataset_score = min(
            finite_sample_scores, key=lambda item: item[1]
        ) if finite_sample_scores else (None, float("nan"))
        candidate = {
            "experiment_id": evaluation["experiment_id"],
            "protocol_id": evaluation["protocol_id"],
            "eligible": not failed_gates,
            "failed_gates": failed_gates,
            "coverage": coverage,
            "runtime_seconds": runtime,
            "recomputed_summary": recomputed,
            "group_summaries": _group_summaries(evaluation["per_sample"]),
            "worst_dataset": worst_dataset,
            "worst_dataset_score": worst_dataset_score,
            **bootstrap,
            "promotion_score": bootstrap["score_ci_low"],
        }
        candidates.append(candidate)
        enriched.append({**evaluation, "recomputed_summary": recomputed})

    candidates.sort(
        key=lambda candidate: (
            candidate["eligible"],
            candidate["promotion_score"] if _finite(candidate["promotion_score"]) else -math.inf,
            candidate["recomputed_summary"]["score"]
            if _finite(candidate["recomputed_summary"]["score"])
            else -math.inf,
            -candidate["runtime_seconds"],
        ),
        reverse=True,
    )
    promoted = False
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        if candidate["eligible"] and not promoted:
            candidate["decision"] = "SUBMIT_CANDIDATE"
            promoted = True
        elif candidate["eligible"]:
            candidate["decision"] = "HOLD"
        else:
            candidate["decision"] = "REJECT"

    comparisons = []
    for left_index in range(len(enriched)):
        for right_index in range(left_index + 1, len(enriched)):
            comparisons.append(
                _paired_delta(
                    enriched[left_index],
                    enriched[right_index],
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + 1000 + left_index * 100 + right_index,
                )
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranking_rule": "official_score_bootstrap_95pct_lower_bound",
        "diagnostics_policy": "hard_gates_only",
        "protocol_id": next(iter(protocols)),
        "dataset_count": len(sample_sets[0]),
        "bootstrap_iterations": bootstrap_iterations,
        "candidates": candidates,
        "paired_comparisons": comparisons,
        "limitations": [
            "Confidence intervals quantify dataset-sampling variability, not split leakage or threshold-selection bias.",
            "A promotion board is globally comparable only when every experiment uses the same frozen, unseen protocol.",
        ],
    }


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}" if _finite(value) else "N/A"


def render_markdown(board: dict[str, Any]) -> str:
    lines = [
        f"# Experiment Promotion Board — {board['protocol_id']}",
        "",
        f"Datasets comparables: `{board['dataset_count']}`. Ranking: límite inferior bootstrap 95% del score oficial.",
        "",
        "| Rank | Experimento | Score oficial | IC 95% | Peor muestra | Adj Edge | Div | Recall | Runtime | Decisión |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in board["candidates"]:
        summary = candidate["recomputed_summary"]
        lines.append(
            "| {rank} | {experiment_id} | {score} | [{low}, {high}] | {worst} | {edge} | "
            "{division} | {recall} | {runtime:.1f}s | {decision} |".format(
                rank=candidate["rank"],
                experiment_id=candidate["experiment_id"],
                score=_format_metric(summary["score"]),
                low=_format_metric(candidate["score_ci_low"]),
                high=_format_metric(candidate["score_ci_high"]),
                worst=_format_metric(candidate["worst_dataset_score"]),
                edge=_format_metric(summary["adj_edge_jaccard"]),
                division=_format_metric(summary["division_jaccard"]),
                recall=_format_metric(summary["node_recall"]),
                runtime=candidate["runtime_seconds"],
                decision=candidate["decision"],
            )
        )
    lines.extend(["", "## Desglose por embrión", ""])
    for candidate in board["candidates"]:
        groups = ", ".join(
            f"{group}={_format_metric(summary['score'])}"
            for group, summary in candidate["group_summaries"].items()
        )
        lines.append(
            f"- `{candidate['experiment_id']}`: {groups}; peor dataset "
            f"`{candidate['worst_dataset']}`={_format_metric(candidate['worst_dataset_score'])}."
        )
    if board["paired_comparisons"]:
        lines.extend(["", "## Comparaciones pareadas", ""])
        for comparison in board["paired_comparisons"]:
            lines.append(
                "- `{left} − {right}`: Δ `{delta}`; IC 95% `[{low}, {high}]`; "
                "P(Δ>0) `{prob}`.".format(
                    left=comparison["left"],
                    right=comparison["right"],
                    delta=_format_metric(comparison["score_delta"]),
                    low=_format_metric(comparison["delta_ci_low"]),
                    high=_format_metric(comparison["delta_ci_high"]),
                    prob=_format_metric(comparison["probability_left_better"]),
                )
            )
    lines.extend(["", "## Limitaciones", ""])
    lines.extend(f"- {limitation}" for limitation in board["limitations"])
    return "\n".join(lines) + "\n"
