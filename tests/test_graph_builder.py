"""
Unit tests for the graph builder search space module.
"""

import unittest
import numpy as np
from core.graph_builder import (
    apply_node_budgeting,
    link_frame_to_frame,
    close_temporal_gaps,
    add_safe_mitotic_divisions,
    build_lineage_graph,
)


class TestGraphBuilder(unittest.TestCase):

    def test_node_budgeting_cap(self):
        # 10 candidate detections in frame 0, estimated_nodes = 5
        dets_by_t = {
            0: [
                {"id": i, "pos": np.array([10.0, 10.0, 10.0]), "conf": i * 0.1}
                for i in range(10)
            ]
        }
        est_nodes = {0: 5}
        # NODE_BUDGET_FACTOR = 1.0 -> should keep top 5 highest confidence
        filtered = apply_node_budgeting(dets_by_t, est_nodes, conf_thresh=0.0, budget_factor=1.0)
        self.assertEqual(len(filtered[0]), 5)
        # Verify highest confidence kept (id 9, 8, 7, 6, 5)
        kept_ids = {c["id"] for c in filtered[0]}
        self.assertIn(9, kept_ids)
        self.assertNotIn(0, kept_ids)

    def test_link_frame_to_frame_lap(self):
        # Frame 0 has 2 cells, Frame 1 has 2 matching cells
        filtered_dets = {
            0: [
                {"id": 1, "pos": np.array([10.0, 20.0, 20.0])},
                {"id": 2, "pos": np.array([30.0, 40.0, 40.0])},
            ],
            1: [
                {"id": 3, "pos": np.array([10.0, 20.2, 20.2])},  # Close to 1
                {"id": 4, "pos": np.array([30.0, 40.2, 40.2])},  # Close to 2
            ],
        }
        edges, fwd, bwd = link_frame_to_frame(filtered_dets, max_dist_um=12.0)
        self.assertEqual(len(edges), 2)
        self.assertEqual(fwd.get(1), 3)
        self.assertEqual(fwd.get(2), 4)

    def test_mitotic_branching_recovery(self):
        # Parent in frame 0 divides into two daughters in frame 1
        filtered_dets = {
            0: [
                {"id": 1, "pos": np.array([10.0, 20.0, 20.0]), "is_mitosis": 0.95},
            ],
            1: [
                {"id": 2, "pos": np.array([10.0, 21.0, 20.0])},  # First daughter
                {"id": 3, "pos": np.array([10.0, 19.0, 20.0])},  # Second daughter
            ],
        }
        # Initial 1:1 LAP link (1 -> 2)
        edges = [(1, 2)]
        fwd = {1: 2}
        bwd = {2: 1}

        # Safe division should attach 3 as second child to 1
        final_edges, mitoses = add_safe_mitotic_divisions(
            filtered_dets, edges, fwd, bwd,
            mitosis_conf_thresh=0.5,
            safe_div_max_um=8.0,
            sister_max_um=12.0,
            frame_cap=0.5,
        )
        self.assertEqual(len(final_edges), 2)
        self.assertIn((1, 2), final_edges)
        self.assertIn((1, 3), final_edges)
        self.assertEqual(len(mitoses), 1)
        self.assertEqual(mitoses[0], (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
