"""Tests for the independent Kaggle public-score leaderboard."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

from kaggle_scores import append_score, best_by_experiment, render_leaderboard, verify_scores  # noqa: E402


class KaggleScoreTests(unittest.TestCase):
    def test_ranking_uses_only_public_score_and_keeps_pending_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiments").mkdir()
            append_score(experiment_id="EXP-101", status="COMPLETE", public_score=0.5, submission_ref=1, source="test", root=root)
            append_score(experiment_id="EXP-102", status="COMPLETE", public_score=0.7, submission_ref=2, source="test", root=root)
            append_score(experiment_id="EXP-103", status="PENDING", public_score=None, submission_ref=3, source="test", root=root)
            # The same pending submission may later acquire a score.
            append_score(experiment_id="EXP-103", status="SCORED", public_score=0.6, submission_ref=3, source="test", root=root)
            self.assertEqual(verify_scores(root), [])
            self.assertEqual(
                [row["experiment_id"] for row in best_by_experiment(root)],
                ["EXP-102", "EXP-103", "EXP-101"],
            )
            view = render_leaderboard(root).read_text(encoding="utf-8")
            self.assertIn("EXP-102", view)
            self.assertIn("0.7000", view)
            self.assertIn("Sin pendientes", view)


if __name__ == "__main__":
    unittest.main()
