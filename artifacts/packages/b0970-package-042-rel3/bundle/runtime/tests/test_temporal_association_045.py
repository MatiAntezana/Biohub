"""Unit tests for Campaign 045 Group-Stratified Asymmetric Birth Ratio Gating (G-ABRG)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_045 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation045Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_045")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["aacg_enabled"])
        self.assertTrue(v1["abrg_enabled"])
        self.assertEqual(v1["min_birth_ratio_44b6"], 0.35)
        self.assertEqual(v1["asym_dist_thresh_44b6_um"], 3.5)
        self.assertEqual(v1["min_birth_ratio_6bba"], 0.22)
        self.assertEqual(v1["asym_dist_thresh_6bba_um"], 4.5)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["aacg_enabled"])
        self.assertTrue(v2["abrg_enabled"])
        self.assertEqual(v2["min_birth_ratio_44b6"], 0.38)
        self.assertEqual(v2["asym_dist_thresh_44b6_um"], 3.2)
        self.assertEqual(v2["min_birth_ratio_6bba"], 0.20)
        self.assertEqual(v2["asym_dist_thresh_6bba_um"], 4.8)

    def test_gabrg_protects_symmetric_cytokinesis(self):
        """Symmetric cytokinesis (r = 1.0) satisfies G-ABRG and is protected in both groups."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0, 10.0],
            [2.0, 10.0, 10.0, 10.0],
            # Daughter 1 (dy = +4 = 1.625 um):
            [3.0, 10.0, 14.0, 10.0],
            [4.0, 10.0, 18.0, 10.0],
            [5.0, 10.0, 22.0, 10.0],
            [6.0, 10.0, 26.0, 10.0],
            [7.0, 10.0, 30.0, 10.0],
            # Daughter 2 (dy = -4 = -1.625 um):
            [3.0, 10.0,  6.0, 10.0],
            [4.0, 10.0,  2.0, 10.0],
            [5.0, 10.0, -2.0, 10.0],
            [6.0, 10.0, -6.0, 10.0],
            [7.0, 10.0, -10.0, 10.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [2, 8], [8, 9], [9, 10], [10, 11], [11, 12]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        for grp in ("44b6", "6bba"):
            res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group=grp)
            self.assertEqual(len(res_edges), len(edges))
            self.assertEqual(audit["forks_pruned"], 0)
            self.assertEqual(audit["abrg_rejected_forks"], 0)

    def test_gabrg_prunes_asymmetric_44b6_fork(self):
        """Asymmetric crossover fork (d1=0.81um, d2=4.06um, r=0.20 < 0.35) gets pruned in 44b6."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 10.0, 20.0, 20.0],
            [2.0, 10.0, 20.0, 20.0],
            # Daughter 1 (continuing mother track: dy = +2 = 0.8125 um):
            [3.0, 10.0, 22.0, 20.0],
            [4.0, 10.0, 24.0, 20.0],
            [5.0, 10.0, 26.0, 20.0],
            [6.0, 10.0, 28.0, 20.0],
            # Daughter 2 (distant entering track: dy = -10 = -4.0625 um, max_d >= 3.5um, r = 0.20):
            [3.0, 10.0, 10.0, 20.0],
            [4.0, 10.0,  8.0, 20.0],
            [5.0, 10.0,  6.0, 20.0],
            [6.0, 10.0,  4.0, 20.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6],
            [2, 7], [7, 8], [8, 9], [9, 10]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="44b6")
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(audit["abrg_rejected_forks"], 1)
        # Verify the continuing mother track (edge 2->3) is retained, spurious edge (2->7) removed:
        self.assertIn((2, 3), set(map(tuple, res_edges)))
        self.assertNotIn((2, 7), set(map(tuple, res_edges)))

    def test_gabrg_preserves_motile_6bba_asymmetry(self):
        """Motile cytokinesis with r=0.30 (d1=1.5um, d2=5.0um) is preserved in 6bba (r >= 0.22)."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # d1 = 3.69 voxels * 0.40625 um = 1.50 um
        # d2 = 12.30 voxels * 0.40625 um = 5.00 um
        # r = 1.50 / 5.00 = 0.30
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 10.0, 20.0, 20.0],
            [2.0, 10.0, 20.0, 20.0],
            # Daughter 1 (dy = +3.69 voxels = 1.50 um, divergent):
            [3.0, 10.0, 23.6923, 20.0],
            [4.0, 10.0, 27.3846, 20.0],
            [5.0, 10.0, 31.0769, 20.0],
            [6.0, 10.0, 34.7692, 20.0],
            # Daughter 2 (dy = -12.3077 voxels = -5.00 um, divergent):
            [3.0, 10.0,  7.6923, 20.0],
            [4.0, 10.0, -4.6154, 20.0],
            [5.0, 10.0, -16.923, 20.0],
            [6.0, 10.0, -29.231, 20.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6],
            [2, 7], [7, 8], [8, 9], [9, 10]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        # In 6bba: min_birth_ratio = 0.22, so r = 0.30 is NOT spurious!
        res_edges_6bba, audit_6bba = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="6bba")
        self.assertEqual(audit_6bba["forks_pruned"], 0)
        self.assertEqual(audit_6bba["abrg_rejected_forks"], 0)
        self.assertEqual(len(res_edges_6bba), len(edges))

        # In 44b6: min_birth_ratio = 0.35, so r = 0.30 IS spurious and pruned!
        res_edges_44b6, audit_44b6 = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="44b6")
        self.assertEqual(audit_44b6["forks_pruned"], 1)
        self.assertEqual(audit_44b6["abrg_rejected_forks"], 1)


if __name__ == "__main__":
    unittest.main()
