"""Unit tests for Campaign 038 Cytokinesis-Enforced Dual-Track Protection (CEDTP)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_038 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation038Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_038")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["guard_requires_cos"])
        self.assertEqual(v1["heading_guard_cos_max"], 0.0)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["guard_requires_cos"])
        self.assertEqual(v2["heading_guard_cos_max"], -0.20)

    def test_cedtp_prunes_parallel_crossover_even_with_long_tracks(self):
        """CEDTP ensures parallel crossovers (cos > 0) are NOT immunized by track persistence."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother m=0 at t=0.
        # Daughter 1 at t=1, continuing to t=5 (length 5).
        # Daughter 2 at t=1, continuing to t=5 (length 5).
        # Both daughters move forward in the same direction: cos_3d > 0 (parallel).
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

        # Under historical 032 (guard_requires_cos=False), len1>=5 and len2>=5 protected the fork:
        prune_cfg_old = {
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
            "heading_guard_cos_max": 0.0,
            "guard_requires_cos": False,
            "photometric_branch_selection": False,
        }
        res_edges_old, audit_old = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg_old)
        # Fork was NOT pruned under old rules:
        self.assertEqual(len(res_edges_old), len(edges))

        # Under CEDTP 038 (guard_requires_cos=True), parallel fork is pruned:
        prune_cfg_038 = dict(prune_cfg_old, guard_requires_cos=True)
        res_edges_new, audit_new = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg_038)
        # Fork MUST be pruned under CEDTP rules!
        self.assertEqual(len(res_edges_new), len(edges) - 1)
        self.assertEqual(audit_new["forks_pruned"], 1)
        # Closer branch [0, 1] survives; jumping branch [0, 6] is pruned:
        self.assertIn([0, 1], res_edges_new.tolist())
        self.assertNotIn([0, 7], res_edges_new.tolist())

    def test_cedtp_protects_diverging_cytokinesis(self):
        """CEDTP protects true cytokinesis divisions with diverging daughter branches (cos <= 0)."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother m=0 at t=0.
        # Branch 1 moves +y, Branch 2 moves -y: cos = -1.0 (antiparallel cytokinesis).
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother
            # Daughter 1 (moves +y):
            [1.0, 10.0, 16.0, 10.0],  # 1
            [2.0, 10.0, 18.0, 10.0],  # 2
            [3.0, 10.0, 20.0, 10.0],  # 3
            # Daughter 2 (moves -y):
            [1.0, 10.0,  4.0, 10.0],  # 4
            [2.0, 10.0,  2.0, 10.0],  # 5
            [3.0, 10.0,  0.0, 10.0],  # 6
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3],
            [0, 4], [4, 5], [5, 6]
        ], dtype=np.int64)

        prune_cfg_038 = {
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
            "heading_guard_cos_max": 0.0,
            "guard_requires_cos": True,
            "photometric_branch_selection": False,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg_038)
        # True cytokinesis fork is 100% preserved!
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)

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
        self.assertLess(smoothed[1, 1], 12.0)
        self.assertGreater(smoothed[1, 1], 10.0)

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
        self.assertIn("smoothing", audit)
        self.assertIn("fork_pruning", audit)


if __name__ == "__main__":
    unittest.main()
