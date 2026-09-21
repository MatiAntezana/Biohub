import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_029 import (
    config,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation029Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_029")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["heading_alignment"])
        self.assertTrue(v1["fork_pruning"]["photometric_branch_selection"])

        v2 = cfg["variants"]["variant_2"]
        self.assertEqual(v2["fork_pruning"]["mode"], "joint_kinematic_photometric")
        self.assertTrue(v2["fork_pruning"]["photometric_asymmetry_filter"])
        self.assertEqual(v2["fork_pruning"]["min_similarity"], 0.20)
        self.assertEqual(v2["fork_pruning"]["min_similarity_margin"], 0.45)

        v3 = cfg["variants"]["variant_3"]
        self.assertFalse(v3["smoothing"]["symmetric_only"])
        self.assertEqual(v3["fork_pruning"]["cos_thresh"], 0.05)

        gates = cfg["gates"]
        self.assertEqual(gates["development_score_delta_min"], 0.003)
        self.assertEqual(gates["confirmation_score_delta_min"], 0.002)

    def test_axial_z_smoothing_symmetric(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 23.0, 34.0],
            [2.0, 10.0, 26.0, 38.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_sm = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = apply_harmonized_axial_regularizer(coords, edges, scale, cfg_sm)
        self.assertEqual(audit["spikes_suppressed"], 1)
        self.assertLess(smoothed[1, 1], coords[1, 1])

    def test_photometric_branch_selection(self):
        # Mother 1 has two daughters: 2 and 3.
        # Daughter 2 has strong descriptor similarity to mother (0.90).
        # Daughter 3 has poor descriptor similarity to mother (0.10).
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 12.0, 10.0],  # 1: mother
            [2.0, 10.0, 15.0, 10.0],  # 2: daughter 1
            [2.0, 10.0, 15.0, 11.0],  # 3: daughter 2
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # Create descriptors: 4-dim normalized unit vectors
        desc = np.zeros((4, 4), dtype=np.float32)
        desc[1] = np.array([1.0, 0.0, 0.0, 0.0])  # Mother: pointing along axis 0
        desc[2] = np.array([0.95, 0.31, 0.0, 0.0])  # Daughter 1: sim ~ 0.95
        desc[3] = np.array([0.10, 0.99, 0.0, 0.0])  # Daughter 2: sim ~ 0.10

        pr_cfg = {
            "enabled": True,
            "mode": "safe_zero_photometric",
            "cos_thresh": 0.00,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "heading_alignment": False,
            "photometric_branch_selection": True,
            "photometric_asymmetry_filter": False
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, desc, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        retained_edges = set(map(tuple, retained))
        self.assertIn((1, 2), retained_edges)
        self.assertNotIn((1, 3), retained_edges)

    def test_photometric_asymmetry_filter(self):
        # Angle is antiparallel (cos_3d = -0.80), but appearance is completely asymmetric
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0
            [1.0, 10.0, 10.0, 10.0],  # 1 (mother)
            [2.0, 10.0, 15.0, 10.0],  # 2 (daughter 1, dy = +5)
            [2.0, 10.0,  5.0, 10.0],  # 3 (daughter 2, dy = -5)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        desc = np.zeros((4, 4), dtype=np.float32)
        desc[1] = np.array([1.0, 0.0, 0.0, 0.0])
        desc[2] = np.array([0.90, 0.43, 0.0, 0.0])  # sim = 0.90
        desc[3] = np.array([0.05, 0.99, 0.0, 0.0])  # sim = 0.05 (severe asymmetry!)

        pr_cfg = {
            "enabled": True,
            "mode": "joint_kinematic_photometric",
            "cos_thresh": 0.00,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "heading_alignment": False,
            "photometric_branch_selection": True,
            "photometric_asymmetry_filter": True,
            "min_similarity": 0.20,
            "min_similarity_margin": 0.45
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, desc, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        retained_edges = set(map(tuple, retained))
        self.assertIn((1, 2), retained_edges)
        self.assertNotIn((1, 3), retained_edges)

    def test_downstream_heading_guard(self):
        # Mother 1 has two outgoing edges: to daughter 2 and daughter 3
        # Daughter 2 continues downstream into track (2 -> 4 -> 5)
        # Daughter 3 is an isolated terminal spur (3 has no children)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0
            [1.0, 10.0, 12.0, 10.0],  # 1 (mother)
            [2.0, 10.0, 14.0, 10.0],  # 2 (continuing daughter)
            [2.0, 10.0, 15.0, 12.0],  # 3 (spurious spur)
            [3.0, 10.0, 16.0, 10.0],  # 4 (child of 2)
            [4.0, 10.0, 18.0, 10.0],  # 5 (grandchild of 2)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [4, 5]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "mode": "safe_zero_photometric",
            "cos_thresh": 0.05,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
            "heading_alignment": True,
            "photometric_branch_selection": False,
            "photometric_asymmetry_filter": False
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        retained_tuples = set(map(tuple, retained))
        self.assertIn((1, 2), retained_tuples)
        self.assertNotIn((1, 3), retained_tuples)

    def test_build_candidate_graph_integrity(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 12.0, 10.0],
            [2.0, 10.0, 14.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        anchor = Lineage(coords, edges).validate()

        cfg = config()
        for vname in ["variant_1", "variant_2", "variant_3"]:
            vcfg = cfg["variants"][vname]
            cand, audit = build_candidate_graph(anchor, scale, None, {}, vcfg, group="44b6")
            self.assertIsInstance(cand, Lineage)
            self.assertEqual(len(cand.coords), len(anchor.coords))

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)


if __name__ == "__main__":
    unittest.main()
