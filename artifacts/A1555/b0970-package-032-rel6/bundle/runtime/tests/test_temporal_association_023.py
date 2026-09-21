import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_023 import (
    config,
    smooth_linear_tracks_guarded_z,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation023Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_023")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["smoothing"]["symmetric_only"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertEqual(v1["fork_pruning"]["cos_thresh"], 0.20)

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["smoothing"]["symmetric_only"])
        self.assertTrue(v2["fork_pruning"]["enabled"])
        self.assertEqual(v2["fork_pruning"]["cos_thresh"], 0.00)

        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["smoothing"]["enabled"])
        self.assertTrue(v3["smoothing"]["symmetric_only"])
        self.assertTrue(v3["fork_pruning"]["enabled"])
        self.assertEqual(v3["fork_pruning"]["cos_thresh"], -0.10)
        self.assertEqual(v3["fork_pruning"]["max_asymm_um"], 8.0)
        self.assertEqual(v3["fork_pruning"]["min_sister_um"], 3.0)

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
        smoothed, audit = smooth_linear_tracks_guarded_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)

        # Lateral XY coordinates must remain strictly untouched
        np.testing.assert_array_equal(smoothed[:, 2:], coords[:, 2:])
        # Z coordinate of node 1 moved closer to z_mid = 10.0
        self.assertLess(smoothed[1, 1], coords[1, 1])
        # Nodes 0 and 2 strictly untouched
        self.assertEqual(smoothed[0, 1], coords[0, 1])
        self.assertEqual(smoothed[2, 1], coords[2, 1])

    def test_cytokinesis_pruning_acute_spurious_fork(self):
        # Mother 1 at t=1, predecessor 0 at t=0 moving along +Y.
        # Daughter 2 continues along +Y (legitimate track).
        # Daughter 3 also moves along +Y with small acute angle (intruder, cos > 0.5).
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
            "cos_thresh": 0.00,
            "max_asymm_um": None,
            "min_sister_um": None,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)

        # Daughter 2 should be kept, edge (1, 3) should be pruned!
        expected_edges = {(0, 1), (1, 2)}
        self.assertEqual(set(map(tuple, retained_edges)), expected_edges)

    def test_cytokinesis_pruning_preserves_true_obtuse_fork(self):
        # Mother 1 at t=1 divides into two daughters moving in OPPOSITE directions (cos < -0.80)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: parent
            [1.0, 10.0, 12.0, 10.0],  # 1: mother
            [2.0, 10.0, 14.0, 10.0],  # 2: daughter 1 (motion: [0, +2, 0])
            [2.0, 10.0, 10.0, 10.0],  # 3: daughter 2 (motion: [0, -2, 0], opposite direction!)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "cos_thresh": 0.00,
            "max_asymm_um": 8.0,
            "min_sister_um": 1.0,
        }
        retained_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(retained_edges), 3)

    def test_build_candidate_graph_invariants(self):
        # 0 -> 1 -> 2 -> (3, 4)
        # Node 1 is an interior linear track with symmetric Z spike (10 -> 12 -> 10)
        # Node 2 is a fork with acute angle (cos > 0.5) to daughters 3 and 4
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0
            [1.0, 12.0, 12.0, 10.0],  # 1: interior linear node with symmetric Z spike
            [2.0, 10.0, 14.0, 10.0],  # 2: mother fork node
            [3.0, 10.0, 16.0, 10.0],  # 3: continuation daughter along +Y
            [3.0, 10.0, 17.0, 11.0],  # 4: acute spur daughter
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        anchor = Lineage(coords, edges).validate()

        cfg = config()
        v2_cfg = cfg["variants"]["variant_2"]
        cand, audit = build_candidate_graph(
            anchor, scale, None, {}, v2_cfg, group="44b6"
        )
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), len(anchor.coords))
        self.assertEqual(len(cand.edges), 3)  # 1 edge pruned from fork (4 -> 3)
        self.assertEqual(audit["fork_pruning"]["forks_pruned"], 1)
        self.assertEqual(audit["smoothing"]["spikes_suppressed"], 1)

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)


if __name__ == "__main__":
    unittest.main()
