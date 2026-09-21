import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_026 import (
    config,
    apply_symmetric_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    recover_antiparallel_cytokinesis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation026Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_026")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertEqual(v1["fork_pruning"]["cos_thresh"], -0.10)
        self.assertEqual(v1["fork_pruning"]["min_ratio"], 0.25)
        self.assertFalse(v1["cytokinesis_recovery"]["enabled"])

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["fork_pruning"]["enabled"])
        self.assertEqual(v2["fork_pruning"]["cos_thresh"], -0.30)
        self.assertEqual(v2["fork_pruning"]["min_ratio"], 0.35)
        self.assertFalse(v2["cytokinesis_recovery"]["enabled"])

        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["cytokinesis_recovery"]["enabled"])
        self.assertEqual(v3["cytokinesis_recovery"]["cos_max"], -0.92)
        self.assertEqual(v3["cytokinesis_recovery"]["min_ratio"], 0.70)

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
        smoothed, audit = apply_symmetric_axial_regularizer(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)
        np.testing.assert_array_equal(smoothed[:, 2:], coords[:, 2:])
        self.assertLess(smoothed[1, 1], coords[1, 1])
        self.assertEqual(smoothed[0, 1], coords[0, 1])
        self.assertEqual(smoothed[2, 1], coords[2, 1])

    def test_cytokinesis_pruning_acute_spurious_fork(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 12.0, 10.0],  # 1: mother (motion: v_in = [0, 2, 0])
            [2.0, 10.0, 14.0, 10.0],  # 2: daughter 1 (motion: v1 = [0, 2, 0], align = 1.0)
            [2.0, 10.0, 15.0, 11.0],  # 3: daughter 2 (motion: v2 = [0, 3, 1], acute angle with v1!)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "cos_thresh": -0.10,
            "min_ratio": 0.25,
            "min_max_dist_um": 3.0,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(retained_edges), 2)
        self.assertIn((1, 2), set(map(tuple, retained_edges)))
        self.assertNotIn((1, 3), set(map(tuple, retained_edges)))

    def test_cytokinesis_pruning_preserves_antiparallel_division(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 10.0, 10.0],  # 1: mother
            [2.0, 10.0,  5.0, 10.0],  # 2: daughter 1 (v1 = [0, -5, 0])
            [2.0, 10.0, 15.0, 10.0],  # 3: daughter 2 (v2 = [0, +5, 0] -> cos = -1.0, ratio = 1.0)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "cos_thresh": -0.10,
            "min_ratio": 0.25,
            "min_max_dist_um": 3.0,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(retained_edges), 3)

    def test_recover_antiparallel_cytokinesis_forks(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: grandmother
            [1.0, 10.0, 10.0, 10.0],  # 1: mother
            [2.0, 10.0,  6.0, 10.0],  # 2: daughter 1 (v1 = [0, -4, 0])
            [3.0, 10.0,  5.0, 10.0],  # 3: d1 continuation
            [4.0, 10.0,  4.0, 10.0],  # 4: d1 continuation 2
            [2.0, 10.0, 14.0, 10.0],  # 5: daughter 2 (v2 = [0, +4, 0], unlinked orphan! cos = -1.0)
            [3.0, 10.0, 15.0, 10.0],  # 6: d2 continuation
            [4.0, 10.0, 16.0, 10.0],  # 7: d2 continuation 2
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7]
        ], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        cyto_cfg = {
            "enabled": True,
            "cos_max": -0.90,
            "min_ratio": 0.70,
            "max_dist_um": 8.0,
            "max_sister_dist_um": 15.0,
            "min_backward_len": 1,
            "min_forward_len": 2,
            "min_desc_corr": 0.0,
        }
        new_edges, audit = recover_antiparallel_cytokinesis_forks(coords, edges, scale, None, cyto_cfg)
        self.assertEqual(audit["forks_recovered"], 1)
        self.assertIn((1, 5), set(map(tuple, new_edges)))

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
