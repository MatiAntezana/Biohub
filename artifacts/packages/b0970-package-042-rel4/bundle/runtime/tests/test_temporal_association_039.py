"""Unit tests for Campaign 039 Calibrated Cytokinesis-Directed Guard (CCDG)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_039 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation039Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_039")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["guard_requires_cos"])
        self.assertEqual(v1["heading_guard_cos_max"], 0.20)
        self.assertEqual(v1["persistence_min_frames"], 2)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["guard_requires_cos"])
        self.assertEqual(v2["heading_guard_cos_max"], 0.25)
        self.assertEqual(v2["persistence_min_frames"], 2)

    def test_ccdg_protects_drift_tilted_division(self):
        """CCDG protects true cytokinesis divisions that are drift-tilted into slightly positive cos (e.g. cos = +0.08)."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother m=0 at t=0.
        # Branch 1 moves +y, Branch 2 moves +y slightly and -z strongly:
        # dot(v1, v2) / (|v1||v2|) = +0.08 (drift-tilted cytokinesis).
        # Both daughters persist for 3 frames (length >= 2).
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother
            # Branch 1:
            [1.0, 10.0, 14.0, 10.0],  # 1 (dy = +4 voxels = 1.625 um)
            [2.0, 10.0, 18.0, 10.0],  # 2
            [3.0, 10.0, 22.0, 10.0],  # 3
            # Branch 2:
            [1.0,  8.0, 10.5, 10.0],  # 4 (dz = -2 voxels = -3.25 um, dy = +0.5 = 0.20 um)
            [2.0,  6.0, 11.0, 10.0],  # 5
            [3.0,  4.0, 11.5, 10.0],  # 6
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3],
            [0, 4], [4, 5], [5, 6]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "spi_max_close_um": 1.5,
            "spi_min_far_um": 2.5,
            "spi_min_ratio": 2.0,
            "distance_cost_mode": "linear",
            "heading_alignment": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "persistence_min_frames": 2,
            "photometric_branch_selection": False,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        # Drift-tilted division is 100% protected!
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)

    def test_ccdg_prunes_acute_parallel_crossover(self):
        """CCDG prunes acute parallel crossovers (cos = 1.0) where cells move in the same direction."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother
            # Branch 1 (closer, d1=1): moves forward +1 in y (length 5 steps forward)
            [1.0, 10.0, 12.0, 10.0],  # 1
            [2.0, 10.0, 14.0, 10.0],  # 2
            [3.0, 10.0, 16.0, 10.0],  # 3
            [4.0, 10.0, 18.0, 10.0],  # 4
            [5.0, 10.0, 20.0, 10.0],  # 5
            [6.0, 10.0, 22.0, 10.0],  # 6
            # Branch 2 (further, d2=7): also moves forward +1 in y, but further away in y (length 5 steps forward)
            [1.0, 10.0, 18.0, 10.0],  # 7
            [2.0, 10.0, 20.0, 10.0],  # 8
            [3.0, 10.0, 22.0, 10.0],  # 9
            [4.0, 10.0, 24.0, 10.0],  # 10
            [5.0, 10.0, 26.0, 10.0],  # 11
            [6.0, 10.0, 28.0, 10.0],  # 12
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "spi_max_close_um": 1.5,
            "spi_min_far_um": 2.5,
            "spi_min_ratio": 2.0,
            "distance_cost_mode": "linear",
            "heading_alignment": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "persistence_min_frames": 2,
            "photometric_branch_selection": False,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        self.assertEqual(len(res_edges), len(edges) - 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertIn([0, 1], res_edges.tolist())
        self.assertNotIn([0, 7], res_edges.tolist())

    def test_axial_z_smoothing_symmetric(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 12.0, 10.0, 10.0],
            [2.0, 10.0, 10.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        sm_cfg = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "max_asymmetry_um": 0.0,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.8,
            "weight_xy": 0.0,
        }
        smoothed, audit = apply_harmonized_axial_regularizer(coords, edges, scale, sm_cfg)
        self.assertEqual(audit["spikes_suppressed"], 1)

    def test_build_candidate_graph_integrity(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 12.0, 10.0],
            [2.0, 10.0, 14.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        lineage = Lineage(coords, edges)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        cfg = config()
        v1_cfg = cfg["variants"]["variant_1"]
        cand, audit = build_candidate_graph(lineage, scale, None, None, v1_cfg, "44b6")
        self.assertEqual(len(cand.coords), len(coords))
        self.assertEqual(len(cand.edges), len(edges))


if __name__ == "__main__":
    unittest.main()
