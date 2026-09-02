"""CPU-only regression tests for the experiment promotion board."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_ROOT))

from promotion_board import build_promotion_board, summarise_rows  # noqa: E402


def metric_row(dataset: str, tp: int, fp: int, fn: int, recall: float) -> dict:
    edge = tp / (tp + fp + fn)
    return {
        "dataset": dataset,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 2,
        "num_pred_nodes": 100,
        "node_recall": recall,
        "total_node_ratio": 0.0,
        "edge_jaccard": edge,
        "adj_edge_jaccard": edge,
    }


def evaluation(experiment_id: str, rows: list[dict], runtime: float = 10.0) -> dict:
    return {
        "status": "SUCCESS",
        "experiment_id": experiment_id,
        "protocol_id": "frozen-v1",
        "requested_dataset_count": len(rows),
        "runtime": {"total_seconds": runtime},
        "per_sample": rows,
        "skipped": [],
    }


class PromotionBoardTests(unittest.TestCase):
    def test_summary_micro_averages_counts(self) -> None:
        rows = [
            metric_row("a", 8, 1, 1, 0.9),
            metric_row("b", 2, 4, 4, 0.5),
        ]
        summary = summarise_rows(rows)
        self.assertAlmostEqual(summary["edge_jaccard"], 10 / 20)
        self.assertEqual(summary["division_jaccard"], 0.0)
        self.assertAlmostEqual(summary["score"], 0.5)

    def test_higher_consistent_model_is_promoted(self) -> None:
        strong = [metric_row(name, 90, 5, 5, 0.95) for name in ("a", "b", "c")]
        weak = [metric_row(name, 70, 15, 15, 0.80) for name in ("a", "b", "c")]
        board = build_promotion_board(
            [evaluation("EXP-STRONG", strong), evaluation("EXP-WEAK", weak)],
            bootstrap_iterations=200,
            bootstrap_seed=7,
        )
        self.assertEqual(board["candidates"][0]["experiment_id"], "EXP-STRONG")
        self.assertEqual(board["candidates"][0]["decision"], "SUBMIT_CANDIDATE")
        self.assertEqual(board["candidates"][0]["worst_dataset"], "a")
        self.assertIn("a", board["candidates"][0]["group_summaries"])
        comparison = board["paired_comparisons"][0]
        self.assertGreater(comparison["delta_ci_low"], 0.0)

    def test_different_sample_sets_are_rejected(self) -> None:
        left = evaluation("LEFT", [metric_row("a", 8, 1, 1, 0.9)])
        right = evaluation("RIGHT", [metric_row("b", 8, 1, 1, 0.9)])
        with self.assertRaisesRegex(ValueError, "same dataset names"):
            build_promotion_board([left, right], bootstrap_iterations=10)

    def test_partial_evaluation_fails_gate(self) -> None:
        candidate = evaluation("PARTIAL", [metric_row("a", 8, 1, 1, 0.9)])
        candidate["requested_dataset_count"] = 2
        board = build_promotion_board([candidate], bootstrap_iterations=10)
        self.assertEqual(board["candidates"][0]["decision"], "REJECT")
        self.assertIn("coverage", board["candidates"][0]["failed_gates"])


if __name__ == "__main__":
    unittest.main()
