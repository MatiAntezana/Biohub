"""
Unit tests for the diagnostic evaluator module.
"""

import unittest
import numpy as np
from core.diagnostic_evaluator import (
    point_distance_um,
    match_nodes_bipartite,
    compute_edge_confusion,
    compute_division_confusion,
    evaluate_tracking,
    DEFAULT_SCALE,
)


class TestDiagnosticEvaluator(unittest.TestCase):

    def test_point_distance_anisotropy(self):
        # 1 voxel displacement in Z: 1.625 um
        p1 = np.array([10.0, 50.0, 50.0])
        p2 = np.array([11.0, 50.0, 50.0])
        d = point_distance_um(p1, p2, DEFAULT_SCALE)
        self.assertAlmostEqual(d, 1.625, places=3)

        # 1 voxel displacement in Y: 0.40625 um
        p3 = np.array([10.0, 51.0, 50.0])
        d_xy = point_distance_um(p1, p3, DEFAULT_SCALE)
        self.assertAlmostEqual(d_xy, 0.40625, places=3)

    def test_node_bipartite_matching(self):
        # Two predicted nodes, two ground truth nodes
        pred_nodes = [
            {"id": 101, "pos": np.array([10.0, 50.0, 50.0])},
            {"id": 102, "pos": np.array([20.0, 80.0, 80.0])},
        ]
        gt_nodes = [
            {"id": 1, "pos": np.array([10.1, 50.1, 50.1])},  # Very close to 101
            {"id": 2, "pos": np.array([50.0, 50.0, 50.0])},  # Far away (> 7 um)
        ]
        p2g, g2p = match_nodes_bipartite(pred_nodes, gt_nodes, DEFAULT_SCALE, max_dist_um=7.0)
        self.assertEqual(p2g.get(101), 1)
        self.assertNotIn(102, p2g)  # Too far from both

    def test_edge_confusion(self):
        # TP, FP, FN
        pred_edges = [(101, 102), (102, 103)]
        gt_edges = [(1, 2), (2, 4)]
        pred_to_gt = {101: 1, 102: 2, 103: 3}
        gt_to_pred = {1: 101, 2: 102, 3: 103}

        # (101, 102) -> (1, 2) is TP
        # (102, 103) -> (2, 3) where in GT node 2 connects to 4, not 3 -> FP
        # GT edge (2, 4) is missing -> FN
        tp, fp, fn = compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_node_penalty_calculation(self):
        # Test penalty when T_pred > T_true
        pred_nodes_by_t = {
            0: [{"id": i, "t": 0, "pos": np.array([10.0, 20.0, 30.0])} for i in range(150)],
        }
        gt_nodes_by_t = {
            0: [{"id": i, "t": 0, "pos": np.array([10.0, 20.0, 30.0])} for i in range(100)],
        }
        res = evaluate_tracking(
            pred_nodes_by_t=pred_nodes_by_t,
            pred_edges=[],
            gt_nodes_by_t=gt_nodes_by_t,
            gt_edges=[],
            estimated_total_nodes=100,
        )
        # T_pred=150, T_true=100 -> delta = 50 / 100 = 0.5 -> penalty = 1 - 0.1 * 0.5 = 0.95
        self.assertAlmostEqual(res.node_penalty_multiplier, 0.95, places=2)


if __name__ == "__main__":
    unittest.main()
