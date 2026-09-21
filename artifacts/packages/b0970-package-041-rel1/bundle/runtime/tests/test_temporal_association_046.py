"""Unit tests for Campaign 046 Tissue-Exclusive Asymmetric Birth Ratio Gating (TE-ABRG)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_046 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation046Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_046")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["aacg_enabled"])
        self.assertTrue(v1["abrg_enabled_44b6"])
        self.assertFalse(v1["abrg_enabled_6bba"])
        self.assertEqual(v1["min_birth_ratio_44b6"], 0.35)
        self.assertEqual(v1["asym_dist_thresh_44b6_um"], 3.5)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["aacg_enabled"])
        self.assertTrue(v2["abrg_enabled_44b6"])
        self.assertFalse(v2["abrg_enabled_6bba"])
        self.assertEqual(v2["min_birth_ratio_44b6"], 0.38)
        self.assertEqual(v2["asym_dist_thresh_44b6_um"], 3.2)

    def test_teabrg_protects_symmetric_cytokinesis(self):
        """Symmetric cytokinesis (r = 1.0) satisfies TE-ABRG and is protected in both groups."""
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

    def test_teabrg_prunes_asymmetric_44b6_fork(self):
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
        self.assertIn((2, 3), set(map(tuple, res_edges)))
        self.assertNotIn((2, 7), set(map(tuple, res_edges)))

    def test_teabrg_disabled_in_6bba(self):
        """In 6bba, ABRG is disabled; an asymmetric divergent cytokinesis fork is preserved."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 10.0, 20.0, 20.0],
            [2.0, 10.0, 20.0, 20.0],
            # Daughter 1 (dy = +2 = 0.8125 um, divergent obtuse):
            [3.0, 10.0, 22.0, 20.0],
            [4.0, 10.0, 24.0, 20.0],
            [5.0, 10.0, 26.0, 20.0],
            [6.0, 10.0, 28.0, 20.0],
            # Daughter 2 (dy = -12 = -4.875 um, max_d >= 4.5um, r = 0.166, divergent obtuse):
            [3.0, 10.0,  8.0, 20.0],
            [4.0, 10.0, -4.0, 20.0],
            [5.0, 10.0, -16.0, 20.0],
            [6.0, 10.0, -28.0, 20.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6],
            [2, 7], [7, 8], [8, 9], [9, 10]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges_6bba, audit_6bba = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="6bba")
        # In 6bba: ABRG is False, obtuse divergent fork satisfies BDPG, so preserved!
        self.assertEqual(audit_6bba["forks_pruned"], 0)
        self.assertEqual(audit_6bba["abrg_rejected_forks"], 0)
        self.assertEqual(len(res_edges_6bba), len(edges))


if __name__ == "__main__":
    unittest.main()
