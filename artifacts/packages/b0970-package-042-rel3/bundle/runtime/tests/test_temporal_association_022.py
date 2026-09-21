import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_022 import (
    config,
    smooth_linear_tracks_guarded_z,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation022Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_022")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["smoothing"]["reversal_only"])
        self.assertTrue(v1["smoothing"]["symmetric_only"])
        self.assertEqual(v1["smoothing"]["max_correction_um"], 0.6)
        self.assertEqual(v1["smoothing"]["jitter_threshold_um"], 0.8)
        self.assertEqual(v1["smoothing"]["weight_z"], 0.80)

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["smoothing"]["reversal_only"])
        self.assertFalse(v2["smoothing"]["symmetric_only"])
        self.assertEqual(v2["smoothing"]["max_frame_reversals"], 4)
        self.assertEqual(v2["smoothing"]["isolated_nn_thresh_um"], 6.5)
        self.assertEqual(v2["smoothing"]["max_correction_um"], 0.6)
        self.assertEqual(v2["smoothing"]["jitter_threshold_um"], 0.8)

        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["smoothing"]["enabled"])
        self.assertTrue(v3["smoothing"]["reversal_only"])
        self.assertFalse(v3["smoothing"]["symmetric_only"])
        self.assertEqual(v3["smoothing"]["max_frame_reversals"], 4)
        self.assertEqual(v3["smoothing"]["isolated_nn_thresh_um"], 5.5)
        self.assertEqual(v3["smoothing"]["max_correction_um"], 0.6)
        self.assertEqual(v3["smoothing"]["jitter_threshold_um"], 0.75)

        gates = cfg["gates"]
        self.assertEqual(gates["development_score_delta_min"], 0.003)
        self.assertEqual(gates["confirmation_score_delta_min"], 0.002)

    def test_axial_z_smoothing_pure_symmetric_invariants(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 23.0, 34.0],  # interior node with symmetric spike in Z
            [2.0, 10.0, 26.0, 38.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)

        # Lateral XY coordinates must remain strictly untouched
        np.testing.assert_array_equal(smoothed[:, 2:], coords[:, 2:])
        # Z coordinate of node 1 moved closer to z_mid = 10.0
        self.assertLess(smoothed[1, 1], coords[1, 1])
        # Nodes 0 and 2 strictly untouched
        self.assertEqual(smoothed[0, 1], coords[0, 1])
        self.assertEqual(smoothed[2, 1], coords[2, 1])

    def test_frame_shock_guard_suppresses_collective_shift(self):
        # Create 5 parallel tracks that all experience an asymmetric Z jump at frame 1
        coords = []
        edges = []
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        for track_idx in range(5):
            base_nid = track_idx * 3
            y_pos = track_idx * 50.0  # far apart so density guard would pass
            coords.extend([
                [0.0, 10.0, y_pos, 0.0],
                [1.0, 13.0, y_pos, 0.0],  # asymmetric reversal jump (10 -> 13 -> 11)
                [2.0, 11.0, y_pos, 0.0],
            ])
            edges.extend([
                [base_nid, base_nid + 1],
                [base_nid + 1, base_nid + 2]
            ])

        coords = np.array(coords, dtype=np.float64)
        edges = np.array(edges, dtype=np.int64)

        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": False,
            "max_frame_reversals": 3,  # threshold is 3, but 5 tracks jumped!
            "isolated_nn_thresh_um": 5.0,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        # All 5 should be suppressed by collective frame shock guard!
        self.assertEqual(audit["frame_shock_suppressed"], 5)
        self.assertEqual(audit["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed, coords)

    def test_density_guard_suppresses_crowded_nodes(self):
        # Two tracks very close to each other at frame 1 (< 3.0 um)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            # Track 1
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 13.0, 20.0, 20.0],  # asymmetric reversal (10 -> 13 -> 11)
            [2.0, 11.0, 20.0, 20.0],
            # Track 2 nearby (dist_y = 1.0 px * 0.40625 = 0.4 um)
            [0.0, 10.0, 21.0, 20.0],
            [1.0, 11.0, 21.0, 20.0],
            [2.0, 11.0, 21.0, 20.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [1, 2],
            [3, 4], [4, 5]
        ], dtype=np.int64)

        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": False,
            "max_frame_reversals": 4,
            "isolated_nn_thresh_um": 6.0,  # neighbor is 0.4 um away!
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["density_suppressed"], 1)
        self.assertEqual(audit["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed, coords)

    def test_isolated_asymmetric_reversal_is_smoothed(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 13.0, 20.0, 20.0],  # asymmetric reversal (10 -> 13 -> 11)
            [2.0, 11.0, 20.0, 20.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": False,
            "max_frame_reversals": 4,
            "isolated_nn_thresh_um": 6.0,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)
        self.assertLess(smoothed[1, 1], coords[1, 1])

    def test_monotonic_z_motion_preserved(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 20.0, 30.0],  # monotonic forward motion (10 -> 12 -> 14)
            [2.0, 14.0, 20.0, 30.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": False,
            "jitter_threshold_um": 0.5,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed, coords)

    def test_build_candidate_graph_validates(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 20.0, 30.0],
            [2.0, 10.0, 20.0, 30.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        anchor = Lineage(coords, edges).validate()

        cfg = config()
        v2_cfg = cfg["variants"]["variant_2"]
        cand, audit = build_candidate_graph(
            anchor, scale, None, {}, v2_cfg, group="44b6"
        )
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), len(anchor.coords))
        self.assertEqual(len(cand.edges), len(anchor.edges))
        self.assertEqual(audit["smoothing"]["spikes_suppressed"], 1)

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)


if __name__ == "__main__":
    unittest.main()
