import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_028 import (
    config,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    recover_additive_cytokinesis_divisions,
    build_candidate_graph,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation028Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_028")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["heading_alignment"])
        self.assertFalse(v1["cytokinesis_recovery"]["enabled"])

        v2 = cfg["variants"]["variant_2"]
        self.assertEqual(v2["fork_pruning"]["mode"], "axial_debiased")
        self.assertEqual(v2["fork_pruning"]["cos_relative_thresh"], -0.10)
        self.assertEqual(v2["fork_pruning"]["cos_xy_thresh"], 0.20)
        self.assertFalse(v2["cytokinesis_recovery"]["enabled"])

        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["cytokinesis_recovery"]["enabled"])
        self.assertEqual(v3["cytokinesis_recovery"]["cos_relative_thresh"], -0.85)
        self.assertEqual(v3["cytokinesis_recovery"]["min_ratio"], 0.50)

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

    def test_axial_debiased_cytokinesis_pruning(self):
        # Two daughters moving in the same lateral direction (c_xy > 0.90)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 12.0, 10.0],  # 1: mother
            [2.0, 10.0, 15.0, 10.0],  # 2: daughter 1 (dy = +3.0)
            [2.0, 10.0, 15.0, 11.0],  # 3: daughter 2 (dy = +3.0, dx = +1.0)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # In XY, both move dy = +3.0, dx = 0.0 -> c_xy = +1.0 (pure parallel motion!)
        pr_cfg = {
            "enabled": True,
            "mode": "axial_debiased",
            "cos_relative_thresh": -0.10,
            "cos_xy_thresh": 0.20,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
            "heading_alignment": True
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(retained), 2)

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
            "mode": "safe_zero_persistence",
            "cos_thresh": 0.05,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
            "heading_alignment": True
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        retained_tuples = set(map(tuple, retained))
        # Daughter 2 MUST be kept because of downstream track persistence and heading
        self.assertIn((1, 2), retained_tuples)
        self.assertNotIn((1, 3), retained_tuples)

    def test_additive_cytokinesis_recovery(self):
        # Mother 1 has linear daughter 2 (v1 = [0, 2, 0])
        # Free candidate 3 at t=2 is completely unlinked (in_deg=0) and moves antiparallel (v2 = [0, -2, 0])
        # Candidate 3 also persists into 4 (out_deg >= 1)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 10.0, 10.0],  # 1: mother
            [2.0, 10.0, 15.0, 10.0],  # 2: daughter 1
            [2.0, 10.0,  5.0, 10.0],  # 3: daughter 2 (free, collinear!)
            [3.0, 10.0,  3.0, 10.0],  # 4: child of 3
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        rec_cfg = {
            "enabled": True,
            "cos_relative_thresh": -0.85,
            "min_ratio": 0.50,
            "min_dist_um": 1.0,
            "max_dist_um": 7.0,
            "require_target_persistence": True
        }
        new_edges, audit = recover_additive_cytokinesis_divisions(coords, edges, scale, None, rec_cfg)
        self.assertEqual(audit["divisions_recovered"], 1)
        self.assertIn((1, 3), set(map(tuple, new_edges)))
        # Verify lineage validity: in_degree <= 1, out_degree <= 2
        cand = Lineage(coords, new_edges).validate()
        self.assertIsInstance(cand, Lineage)

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
