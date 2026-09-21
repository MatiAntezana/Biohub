import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_024 import (
    config,
    apply_symmetric_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    recover_antiparallel_cytokinesis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation024Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_024")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["smoothing"]["symmetric_only"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertEqual(v1["fork_pruning"]["cos_thresh"], 0.20)
        self.assertEqual(v1["fork_pruning"]["min_ratio"], 0.15)
        self.assertFalse(v1["cytokinesis_recovery"]["enabled"])

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["fork_pruning"]["enabled"])
        self.assertEqual(v2["fork_pruning"]["cos_thresh"], 0.10)
        self.assertTrue(v2["fork_pruning"]["persistence_guard"])
        self.assertFalse(v2["cytokinesis_recovery"]["enabled"])

        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["smoothing"]["enabled"])
        self.assertTrue(v3["fork_pruning"]["enabled"])
        self.assertTrue(v3["cytokinesis_recovery"]["enabled"])
        self.assertEqual(v3["cytokinesis_recovery"]["cos_max"], -0.85)
        self.assertEqual(v3["cytokinesis_recovery"]["min_ratio"], 0.35)

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

        # Lateral XY coordinates must remain strictly untouched
        np.testing.assert_array_equal(smoothed[:, 2:], coords[:, 2:])
        # Z coordinate of node 1 moved closer to z_mid = 10.0
        self.assertLess(smoothed[1, 1], coords[1, 1])
        # Nodes 0 and 2 strictly untouched
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
            "cos_thresh": 0.20,
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
            "persistence_guard": False,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(retained_edges), 2)
        # Legitimate edge (1, 2) kept; intruder (1, 3) removed
        self.assertIn((1, 2), set(map(tuple, retained_edges)))
        self.assertNotIn((1, 3), set(map(tuple, retained_edges)))

    def test_cytokinesis_pruning_extreme_asymmetry(self):
        # Mother 0 at t=0. Daughter 1 is 0.5 um away (true continuation).
        # Daughter 2 is 8.0 um away (passing outlier). Ratio = 0.5 / 8.0 = 0.0625 (< 0.15).
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother
            [1.0, 10.0, 10.5, 10.0],  # 1: daughter 1 (0.5 um)
            [1.0, 10.0, 18.0, 10.0],  # 2: daughter 2 (8.0 um)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "cos_thresh": 0.50,  # cos is +1.0, but even if cos was negative:
            "min_ratio": 0.15,
            "min_max_dist_um": 3.0,
            "persistence_guard": False,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertIn((0, 1), set(map(tuple, retained_edges)))
        self.assertNotIn((0, 2), set(map(tuple, retained_edges)))

    def test_cytokinesis_recovery_antiparallel_pair(self):
        # Mother track from t=0..10.
        # At t=10, mother has child d1 at t=11 moving in +Y direction.
        # At t=11, orphan d2 moving in -Y direction (antiparallel, cos = -1.0).
        # d2 track continues from t=11..22.
        coords_list = []
        edges_list = []
        # Mother track 0..10
        for t in range(11):
            coords_list.append([float(t), 10.0, 10.0 + t*0.5, 10.0])
            if t > 0:
                edges_list.append([t-1, t])
        # Daughter 1 at t=11 (node 11)
        coords_list.append([11.0, 10.0, 15.0 + 3.0, 10.0])  # v1 = [0, +3.0, 0]
        edges_list.append([10, 11])
        # Daughter 2 orphan at t=11 (node 12)
        coords_list.append([11.0, 10.0, 15.0 - 3.0, 10.0])  # v2 = [0, -3.0, 0] (antiparallel!)
        # Continue d2 track for 11 frames (t=12..22)
        for step in range(1, 12):
            coords_list.append([11.0 + step, 10.0, 12.0 - step*0.5, 10.0])
            edges_list.append([12 + step - 1, 12 + step])

        coords = np.array(coords_list, dtype=np.float64)
        edges = np.array(edges_list, dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        cyto_cfg = {
            "enabled": True,
            "cos_max": -0.85,
            "min_ratio": 0.35,
            "max_dist_um": 8.0,
            "max_sister_dist_um": 12.0,
            "min_backward_len": 10,
            "min_forward_len": 10,
            "min_desc_corr": 0.40,
        }
        # Descriptors: dummy unit-norm vectors with dot product = 1.0
        descriptors = np.ones((len(coords), 10), dtype=np.float32)

        new_edges, audit = recover_antiparallel_cytokinesis_forks(coords, edges, scale, descriptors, cyto_cfg)
        self.assertEqual(audit["candidates_evaluated"], 1)
        self.assertEqual(audit["forks_recovered"], 1)
        self.assertIn((10, 12), set(map(tuple, new_edges)))

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
