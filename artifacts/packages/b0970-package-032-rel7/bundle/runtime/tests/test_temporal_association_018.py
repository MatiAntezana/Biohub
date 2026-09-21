import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_018 import (
    config,
    smooth_linear_tracks_conditional_z,
    bridge_single_frame_gaps,
    build_candidate_graph,
    evaluate_association_panel,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation018Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_018")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])
        self.assertTrue(cfg["variants"]["variant_1"]["smoothing"]["enabled"])
        self.assertFalse(cfg["variants"]["variant_1"]["gaps"]["enabled"])
        self.assertIn("smoothing", cfg["variants"]["variant_3"])
        self.assertIn("gaps", cfg["variants"]["variant_3"])

    def test_axial_z_smoothing_leaves_xy_strictly_untouched(self):
        # Linear track: t=0, t=1, t=2
        # Center node has curved XY and jitter in Z (spike of 3.0 voxels = 4.875 um)
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 14.0, 23.0, 34.0],  # interior node with jitter in Z and curve in XY
            [2.0, 12.0, 28.0, 40.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_smooth = {
            "enabled": True,
            "jitter_threshold_um": 1.2,
            "max_correction_um": 0.8,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)

        # XY coordinates must remain exactly identical
        np.testing.assert_array_equal(smoothed[:, 2], coords[:, 2])
        np.testing.assert_array_equal(smoothed[:, 3], coords[:, 3])

        # Z coordinate of node 1 must be adjusted towards midpoint (11.0)
        orig_z = coords[1, 1]
        smooth_z = smoothed[1, 1]
        self.assertLess(smooth_z, orig_z)
        self.assertGreater(smooth_z, 10.0)

        # Endpoints must remain unchanged
        self.assertEqual(smoothed[0, 1], coords[0, 1])
        self.assertEqual(smoothed[2, 1], coords[2, 1])

    def test_monotonic_z_motion_preserved_by_reversal_guard(self):
        # Linear track moving monotonically in Z: t=0 (z=10), t=1 (z=14), t=2 (z=20)
        # acc_z_um = |14 - 15| * 1.625 = 1.625 um > 1.2 um
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 14.0, 20.0, 30.0],
            [2.0, 20.0, 20.0, 30.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # Case A: reversal_only=False -> smoothed
        cfg_no_guard = {
            "enabled": True,
            "reversal_only": False,
            "jitter_threshold_um": 1.2,
            "max_correction_um": 0.8,
            "weight_z": 0.80,
        }
        _, audit_a = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_no_guard)
        self.assertEqual(audit_a["spikes_suppressed"], 1)

        # Case B: reversal_only=True -> NOT smoothed (monotonic velocity preserved)
        cfg_guarded = {
            "enabled": True,
            "reversal_only": True,
            "jitter_threshold_um": 1.2,
            "max_correction_um": 0.8,
            "weight_z": 0.80,
        }
        smoothed_b, audit_b = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_guarded)
        self.assertEqual(audit_b["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed_b, coords)

    def test_gap_bridging_inserts_interpolated_node_and_edges(self):
        # Track 1 ends at t=0: node 0 -> node 1
        # Track 2 starts at t=2: node 2 -> node 3
        # Missing detection at t=1
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0
            [0.0, 10.2, 20.2, 30.2],  # 1 (end of track 1 at t=0, wait t=0 -> 1)
            [2.0, 10.8, 20.8, 30.8],  # 2 (start of track 2 at t=2)
            [3.0, 11.0, 21.0, 31.0],  # 3
        ], dtype=np.float64)
        # Fix frames: 0 at t=0, 1 at t=1... wait:
        # We need end at t, start at t+2.
        coords[0, 0] = 0.0
        coords[1, 0] = 1.0  # ends at t=1
        coords[2, 0] = 3.0  # starts at t=3
        coords[3, 0] = 4.0

        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # High similarity descriptors between node 1 and node 2
        descriptors = np.zeros((4, 405), dtype=np.float32)
        base = np.random.RandomState(42).randn(405).astype(np.float32)
        base /= np.linalg.norm(base)
        descriptors[1] = base
        descriptors[2] = base + 0.01 * np.random.RandomState(43).randn(405).astype(np.float32)
        descriptors[2] /= np.linalg.norm(descriptors[2])

        cfg_gaps = {
            "enabled": True,
            "max_distance_um": 4.5,
            "min_similarity": 0.60,
            "spatial_exclusion_um": 2.0
        }

        new_coords, new_edges, audit = bridge_single_frame_gaps(
            coords, edges, scale, descriptors, cfg_gaps
        )
        self.assertEqual(audit["gaps_bridged"], 1)
        self.assertEqual(audit["nodes_added"], 1)
        self.assertEqual(audit["edges_added"], 2)
        self.assertEqual(len(new_coords), 5)
        self.assertEqual(len(new_edges), 4)

        # Rescued node is node 4 at t=2
        rescued = new_coords[4]
        self.assertEqual(rescued[0], 2.0)  # frame t=2
        # Midpoint of node 1 (10.2, 20.2, 30.2) and node 2 (10.8, 20.8, 30.8)
        self.assertAlmostEqual(rescued[1], 10.5)
        self.assertAlmostEqual(rescued[2], 20.5)
        self.assertAlmostEqual(rescued[3], 30.5)

        # Edges include 1 -> 4 and 4 -> 2
        edge_set = {tuple(e) for e in new_edges}
        self.assertIn((1, 4), edge_set)
        self.assertIn((4, 2), edge_set)

    def test_gap_bridging_respects_spatial_exclusion(self):
        # Same setup as above, but with an existing node at t=2 right at the midpoint
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0
            [1.0, 10.2, 20.2, 30.2],  # 1 (end at t=1)
            [3.0, 10.8, 20.8, 30.8],  # 2 (start at t=3)
            [4.0, 11.0, 21.0, 31.0],  # 3
            [2.0, 10.5, 20.5, 30.5],  # 4: existing node at t=2 right at midpoint!
        ], dtype=np.float64)
        edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        descriptors = np.zeros((5, 405), dtype=np.float32)
        base = np.random.RandomState(42).randn(405).astype(np.float32)
        base /= np.linalg.norm(base)
        descriptors[1] = base
        descriptors[2] = base

        cfg_gaps = {
            "enabled": True,
            "max_distance_um": 4.5,
            "min_similarity": 0.60,
            "spatial_exclusion_um": 2.0
        }

        new_coords, new_edges, audit = bridge_single_frame_gaps(
            coords, edges, scale, descriptors, cfg_gaps
        )
        # Should NOT bridge due to collision with node 4
        self.assertEqual(audit["gaps_bridged"], 0)
        self.assertEqual(len(new_coords), 5)
        self.assertEqual(len(new_edges), 2)

    def test_build_candidate_graph_validates(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 10.5, 20.5, 30.5],
            [2.0, 11.0, 21.0, 31.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        descriptors = np.zeros((3, 405), dtype=np.float32)
        stats = np.ones((3, 3), dtype=np.float32)

        anchor = Lineage(coords, edges).validate()
        variant_cfg = {
            "smoothing": {"enabled": True, "jitter_threshold_um": 1.2, "max_correction_um": 0.8, "weight_z": 0.80, "weight_xy": 0.0},
            "gaps": {"enabled": False}
        }
        cand, audit = build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg)
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), 3)
        self.assertEqual(len(cand.edges), 2)


if __name__ == "__main__":
    unittest.main()
