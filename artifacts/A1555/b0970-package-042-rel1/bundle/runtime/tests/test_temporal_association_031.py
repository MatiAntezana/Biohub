"""Unit tests for Campaign 031 Track-Length Persistence Guarded Cytokinesis Regularization."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_031 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation031Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_031")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])
        self.assertIn("gates", cfg)

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
        }
        new_coords, audit = apply_harmonized_axial_regularizer(coords, edges, scale, sm_cfg)
        self.assertEqual(audit["spikes_suppressed"], 1)
        self.assertLess(new_coords[1, 1], coords[1, 1])

    def test_track_length_persistence_guard_prunes_short_spur(self):
        # Mother at t=0, d1 at t=1 continuing to t=5 (L=4), d2 at t=1 terminating at t=1 (L=0)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0], # 0: mother
            [1.0, 10.0, 10.5, 10.0], # 1: d1
            [1.0, 10.0, 11.0, 10.0], # 2: d2 (acute divergence, cos > 0)
            [2.0, 10.0, 10.8, 10.0], # 3: d1 child
            [3.0, 10.0, 11.1, 10.0], # 4: d1 grandchild
            [4.0, 10.0, 11.4, 10.0], # 5: d1 great-grandchild
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [0, 2],
            [1, 3], [3, 4], [4, 5]
        ], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        pr_cfg = {
            "enabled": True,
            "mode": "track_length_persistence_guarded",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "heading_alignment": True,
            "photometric_branch_selection": False,
            "max_prune_track_length": 2,
            "length_asymmetry_guard": False,
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        # Check that d2 was pruned, d1 was kept
        self.assertNotIn((0, 2), [(int(u), int(v)) for u, v in retained])
        self.assertIn((0, 1), [(int(u), int(v)) for u, v in retained])

    def test_track_length_persistence_guard_preserves_persistent_fork(self):
        # Mother at t=0, both d1 and d2 continue for 4 frames (L=3)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0], # 0: mother
            [1.0, 10.0, 10.5, 10.0], # 1: d1
            [1.0, 10.0, 11.0, 10.0], # 2: d2
            [2.0, 10.0, 10.8, 10.0], # 3: d1 child
            [3.0, 10.0, 11.1, 10.0], # 4: d1 grandchild
            [4.0, 10.0, 11.4, 10.0], # 5: d1 great-grandchild
            [2.0, 10.0, 11.5, 10.0], # 6: d2 child
            [3.0, 10.0, 12.0, 10.0], # 7: d2 grandchild
            [4.0, 10.0, 12.5, 10.0], # 8: d2 great-grandchild
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [0, 2],
            [1, 3], [3, 4], [4, 5],
            [2, 6], [6, 7], [7, 8]
        ], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        pr_cfg = {
            "enabled": True,
            "mode": "track_length_persistence_guarded",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "heading_alignment": True,
            "photometric_branch_selection": False,
            "max_prune_track_length": 2,
            "length_asymmetry_guard": False,
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        # BOTH ARE PERSISTENT (L=3 > 2): ZERO FORKS PRUNED!
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(retained), len(edges))

    def test_asymmetry_length_guard(self):
        # d1 continues for 10 frames (L=9), d2 continues for 3 frames (L=3).
        # Under V1 (max_len=2), d2 is kept.
        # Under V2 (asymmetry guard with min_asym=3, max_asym_p=3), d2 is pruned because 9-3 >= 3!
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0], # 0
            [1.0, 10.0, 10.5, 10.0], # 1
            [1.0, 10.0, 11.0, 10.0], # 2 (v2 = [0, 1.0, 0])
            [2.0, 10.0, 10.8, 10.0], # 3
            [3.0, 10.0, 11.1, 10.0], # 4
            [4.0, 10.0, 11.4, 10.0], # 5
            [5.0, 10.0, 11.7, 10.0], # 6
            [6.0, 10.0, 12.0, 10.0], # 7
            [7.0, 10.0, 12.3, 10.0], # 8
            [8.0, 10.0, 12.6, 10.0], # 9
            [2.0, 10.0, 11.0, 11.0], # 10 (d2 child turns along x, w2 = [0, 0, 1.0], head2 = 0.0)
            [3.0, 10.0, 11.0, 12.0], # 11
            [4.0, 10.0, 11.0, 13.0], # 12 (d2 length = 3)
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [0, 2],
            [1, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
            [2, 10], [10, 11], [11, 12]
        ], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        
        # Test V1: L=3 > 2 -> not pruned
        pr_cfg_v1 = {
            "enabled": True,
            "mode": "track_length_persistence_guarded",
            "cos_thresh": 0.0,
            "max_prune_track_length": 2,
            "length_asymmetry_guard": False,
        }
        ret_v1, aud_v1 = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg_v1)
        self.assertEqual(aud_v1["forks_pruned"], 0)

        # Test V2: L=3 and delta_L = 8-3 = 5 >= 3 -> pruned!
        pr_cfg_v2 = {
            "enabled": True,
            "mode": "track_length_persistence_guarded",
            "cos_thresh": 0.0,
            "max_prune_track_length": 2,
            "length_asymmetry_guard": True,
            "max_asym_prune_length": 3,
            "length_asymmetry_min": 3,
        }
        ret_v2, aud_v2 = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg_v2)
        self.assertEqual(aud_v2["forks_pruned"], 1)

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
        vcfg = cfg["variants"]["variant_1"]
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
