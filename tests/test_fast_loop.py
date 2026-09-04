"""
Integration tests for the Fast-Loop end-to-end execution.
"""

import unittest
from fast_loop.evaluate_fast import run_fast_evaluation
from fast_loop.orchestrator import run_experiment_cycle


class TestFastLoopIntegration(unittest.TestCase):

    def test_run_fast_evaluation_metrics_contract(self):
        res = run_fast_evaluation(quiet=True)
        # Verify required keys in telemetry contract
        required_keys = [
            "total_score",
            "adjusted_edge_jaccard",
            "division_jaccard",
            "unadjusted_edge_jaccard",
            "node_penalty_multiplier",
            "node_recall_7um",
            "total_predicted_nodes",
            "estimated_true_nodes",
            "edge_tp", "edge_fp", "edge_fn",
            "div_tp", "div_fp", "div_fn",
            "edges_recovered", "edges_fragmented",
            "edges_lost_to_detection", "wrong_association_edges",
            "timing",
        ]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertGreater(res["total_score"], 0.0)
        self.assertGreater(res["node_recall_7um"], 0.8)
        self.assertLessEqual(res["timing"]["total_wall_seconds"], 5.0)

    def test_orchestrator_dry_cycle(self):
        # Run an experiment cycle with no-git flag
        res = run_experiment_cycle(
            hypothesis="Integration test hypothesis",
            target_metric="total_score",
            auto_git=False,
        )
        self.assertIn("score", res)
        self.assertIn("run_id", res)


if __name__ == "__main__":
    unittest.main()
