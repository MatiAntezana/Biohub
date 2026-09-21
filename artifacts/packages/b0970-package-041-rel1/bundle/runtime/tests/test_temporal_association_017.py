import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_017 import (
    config,
    prune_spurious_forks,
    smooth_linear_tracks_conditional_z,
    build_candidate_graph,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation017Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_017")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("fork_pruning", cfg["variants"]["variant_1"])
        self.assertTrue(cfg["variants"]["variant_1"]["fork_pruning"]["enabled"])
        self.assertFalse(cfg["variants"]["variant_1"]["smoothing"]["enabled"])
        self.assertTrue(cfg["variants"]["variant_2"]["smoothing"]["enabled"])
        self.assertTrue(cfg["variants"]["variant_2"]["smoothing"]["conditional_jitter"])

    def test_genuine_mitosis_preserved_by_invariants(self):
        # 1 mother at t=0, 2 symmetric daughters at t=1
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0: mother
            [1.0, 10.5, 21.0, 31.0],  # 1: daughter 1
            [1.0,  9.5, 19.0, 29.0],  # 2: daughter 2
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # High mutual resemblance and sister resemblance
        descriptors = np.zeros((3, 405), dtype=np.float32)
        base = np.random.RandomState(42).randn(405).astype(np.float32)
        base /= np.linalg.norm(base)
        descriptors[0] = base
        descriptors[1] = base + 0.02 * np.random.RandomState(43).randn(405).astype(np.float32)
        descriptors[1] /= np.linalg.norm(descriptors[1])
        descriptors[2] = base + 0.02 * np.random.RandomState(44).randn(405).astype(np.float32)
        descriptors[2] /= np.linalg.norm(descriptors[2])
        stats = np.ones((3, 3), dtype=np.float32)

        cfg_prune = {
            "enabled": True,
            "min_mother_dot": 0.45,
            "min_sister_dot": 0.35,
            "max_dist_um": 4.8,
            "max_sister_sep_um": 7.0,
            "max_dist_diff_um": 2.5,
        }
        pruned_edges, audit = prune_spurious_forks(coords, edges, scale, descriptors, stats, cfg_prune)
        self.assertEqual(audit["forks_certified"], 1)
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(pruned_edges), 2)

    def test_spurious_fork_imposter_pruned(self):
        # 1 mother at t=0, 1 real daughter close with high correlation, 1 imposter far with low correlation
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0: mother
            [1.0, 10.2, 20.2, 30.2],  # 1: real daughter (close)
            [1.0, 18.0, 25.0, 35.0],  # 2: imposter (far)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        descriptors = np.zeros((3, 405), dtype=np.float32)
        v0 = np.random.RandomState(1).randn(405).astype(np.float32); v0 /= np.linalg.norm(v0)
        v1 = v0 + 0.05 * np.random.RandomState(2).randn(405).astype(np.float32); v1 /= np.linalg.norm(v1)
        v2 = np.random.RandomState(3).randn(405).astype(np.float32); v2 /= np.linalg.norm(v2)
        descriptors[0] = v0
        descriptors[1] = v1
        descriptors[2] = v2
        stats = np.ones((3, 3), dtype=np.float32)

        cfg_prune = {
            "enabled": True,
            "min_mother_dot": 0.45,
            "min_sister_dot": 0.35,
            "max_dist_um": 4.8,
            "max_sister_sep_um": 7.0,
            "max_dist_diff_um": 2.5,
        }
        pruned_edges, audit = prune_spurious_forks(coords, edges, scale, descriptors, stats, cfg_prune)
        self.assertEqual(audit["forks_certified"], 0)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(pruned_edges), 1)
        # Verify that the kept edge is the real daughter (0 -> 1)
        self.assertEqual(tuple(pruned_edges[0]), (0, 1))

    def test_conditional_z_smoothing_suppresses_spike_without_touching_xy(self):
        # Linear track: t=0, t=1, t=2
        # Node 1 has an anomalous Z spike (+4.0 voxels = 6.5 um), but smooth XY
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 14.0, 21.0, 31.0],  # spike in Z
            [2.0, 10.0, 22.0, 32.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_smoothing = {
            "enabled": True,
            "conditional_jitter": True,
            "jitter_threshold_um": 2.5,
            "max_correction_um": 1.5,
            "weight_z": 0.85,
            "weight_xy": 0.0,
        }
        smoothed, audit = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smoothing)
        self.assertEqual(audit["spikes_suppressed"], 1)
        # Z should be reduced towards 10.0
        self.assertLess(smoothed[1, 1], coords[1, 1])
        # XY must remain 100% UNCHANGED
        self.assertEqual(smoothed[1, 2], coords[1, 2])
        self.assertEqual(smoothed[1, 3], coords[1, 3])

    def test_smooth_path_untouched(self):
        # Linear track with perfectly smooth motion: no Z spike
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 10.2, 21.0, 31.0],  # normal displacement
            [2.0, 10.4, 22.0, 32.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_smoothing = {
            "enabled": True,
            "conditional_jitter": True,
            "jitter_threshold_um": 2.5,
            "max_correction_um": 1.5,
            "weight_z": 0.85,
            "weight_xy": 0.0,
        }
        smoothed, audit = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smoothing)
        self.assertEqual(audit["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed, coords)


if __name__ == "__main__":
    unittest.main()
