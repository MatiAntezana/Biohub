import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_027 import (
    config,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation027Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_027")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["smoothing"]["symmetric_only"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertEqual(v1["fork_pruning"]["cos_thresh"], 0.05)
        self.assertEqual(v1["fork_pruning"]["min_ratio"], 0.15)

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertFalse(v2["smoothing"]["symmetric_only"])
        self.assertEqual(v2["smoothing"]["max_asymmetry_um"], 0.8)

        v3 = cfg["variants"]["variant_3"]
        self.assertEqual(v3["fork_pruning"]["cos_thresh"], 0.00)
        self.assertEqual(v3["fork_pruning"]["min_ratio"], 0.18)

        gates = cfg["gates"]
        self.assertEqual(gates["development_score_delta_min"], 0.003)
        self.assertEqual(gates["confirmation_score_delta_min"], 0.002)

    def test_axial_z_smoothing_quasi_symmetric(self):
        # Interior node with quasi-symmetric spike in Z (differs by 0.3 slices = 0.4875 um <= 0.8 um)
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 23.0, 34.0],
            [2.0, 10.3, 26.0, 38.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # Pure symmetric rejects
        cfg_pure = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed_pure, audit_pure = apply_harmonized_axial_regularizer(coords, edges, scale, cfg_pure)
        self.assertEqual(audit_pure["spikes_suppressed"], 0)

        # Quasi-symmetric accepts
        cfg_quasi = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": False,
            "max_asymmetry_um": 0.8,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed_quasi, audit_quasi = apply_harmonized_axial_regularizer(coords, edges, scale, cfg_quasi)
        self.assertEqual(audit_quasi["spikes_suppressed"], 1)
        self.assertLess(smoothed_quasi[1, 1], coords[1, 1])

    def test_cytokinesis_pruning_safe_zero_preserves_tp2(self):
        # TP 2 geometry: cos = -0.0205, ratio = 0.4311
        # Should be preserved by cos_thresh = 0.05 and cos_thresh = 0.00!
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 10.0, 10.0],  # 1: mother
            [2.0, 10.0,  8.0, 10.0],  # 2: daughter 1 (v1 = [0, -2, 0])
            [2.0, 10.0, 10.0, 14.0],  # 3: daughter 2 (v2 = [0, 0, +4] -> cos = 0.0, orthogonal!)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # cos_thresh = 0.05 preserves orthogonal forks
        pr_cfg = {
            "enabled": True,
            "cos_thresh": 0.05,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(retained_edges), 3)

    def test_cytokinesis_pruning_prunes_acute_fork(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 12.0, 10.0],  # 1: mother (v_in = [0, 2, 0])
            [2.0, 10.0, 14.0, 10.0],  # 2: daughter 1 (v1 = [0, 2, 0], align = 1.0)
            [2.0, 10.0, 15.0, 11.0],  # 3: daughter 2 (v2 = [0, 3, 1], acute angle cos = +0.94!)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "cos_thresh": 0.05,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(retained_edges), 2)
        self.assertIn((1, 2), set(map(tuple, retained_edges)))

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
        v1_cfg = cfg["variants"]["variant_1"]
        cand, audit = build_candidate_graph(anchor, scale, None, {}, v1_cfg, group="44b6")
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), len(anchor.coords))
        self.assertEqual(len(cand.edges), len(anchor.edges))

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)


if __name__ == "__main__":
    unittest.main()
