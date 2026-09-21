import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_025 import (
    config,
    apply_symmetric_axial_regularizer,
    untangle_crossover_swaps,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation025Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_025")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])

        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["fork_pruning"]["enabled"])
        self.assertTrue(v1["crossover_untangling"]["enabled"])
        self.assertEqual(v1["crossover_untangling"]["min_delta_momentum"], 1.20)
        self.assertEqual(v1["crossover_untangling"]["max_cur_cos"], 0.10)

        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["fork_pruning"]["enabled"])
        self.assertTrue(v2["crossover_untangling"]["enabled"])
        self.assertTrue(v2["crossover_untangling"]["use_descriptors"])

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

    def test_crossover_untangling_hairpin_swap(self):
        # Two tracks crossing at t=1:
        # Track 1: p1(0, 10, 0, 0) -> u1(1, 10, 2, 0) -> erroneously linked to v1(2, 10, 1, 0) -> s1(3, 10, 0, 0)
        # Track 2: p2(0, 10, 4, 0) -> u2(1, 10, 2.5, 0) -> erroneously linked to v2(2, 10, 3.5, 0) -> s2(3, 10, 4.5, 0)
        coords = np.array([
            [0.0, 10.0, 0.0, 0.0],    # 0: p1
            [1.0, 10.0, 2.0, 0.0],    # 1: u1 (v_in1 = [0, +2, 0])
            [2.0, 10.0, 1.0, 0.0],    # 2: v1 (v_out1 = [0, -1, 0] -> cos = -1.0!)
            [3.0, 10.0, 0.0, 0.0],    # 3: s1
            [0.0, 10.0, 4.5, 0.0],    # 4: p2
            [1.0, 10.0, 2.5, 0.0],    # 5: u2 (v_in2 = [0, -2, 0])
            [2.0, 10.0, 3.5, 0.0],    # 6: v2 (v_out2 = [0, +1, 0] -> cos = -1.0!)
            [3.0, 10.0, 4.5, 0.0],    # 7: s2
        ], dtype=np.float64)

        # Erroneous edges with hairpin turns:
        edges = np.array([
            [0, 1], [1, 2], [2, 3],
            [4, 5], [5, 6], [6, 7],
        ], dtype=np.int64)

        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        xo_cfg = {
            "enabled": True,
            "max_spatial_dist_um": 5.5,
            "min_delta_momentum": 1.20,
            "max_cur_cos": 0.10,
            "min_swap_cos": 0.60,
            "max_swaps_per_dataset": 50,
            "use_descriptors": False,
        }

        new_edges, audit = untangle_crossover_swaps(coords, edges, scale, None, xo_cfg)
        self.assertEqual(audit["crossovers_inspected"], 1)
        self.assertEqual(audit["crossovers_untangled"], 1)

        # Edges (1, 2) and (5, 6) must be swapped to (1, 6) and (5, 2)!
        new_edge_set = set(map(tuple, new_edges))
        self.assertNotIn((1, 2), new_edge_set)
        self.assertNotIn((5, 6), new_edge_set)
        self.assertIn((1, 6), new_edge_set)
        self.assertIn((5, 2), new_edge_set)

    def test_crossover_untangling_preserves_clean_parallel_tracks(self):
        # Two clean parallel tracks moving forward:
        coords = np.array([
            [0.0, 10.0, 0.0, 0.0],   # 0: p1
            [1.0, 10.0, 2.0, 0.0],   # 1: u1
            [2.0, 10.0, 4.0, 0.0],   # 2: v1 (cos = +1.0)
            [3.0, 10.0, 6.0, 0.0],   # 3: s1
            [0.0, 10.0, 1.0, 5.0],   # 4: p2
            [1.0, 10.0, 3.0, 5.0],   # 5: u2
            [2.0, 10.0, 5.0, 5.0],   # 6: v2 (cos = +1.0)
            [3.0, 10.0, 7.0, 5.0],   # 7: s2
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [1, 2], [2, 3],
            [4, 5], [5, 6], [6, 7],
        ], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        xo_cfg = {
            "enabled": True,
            "max_spatial_dist_um": 6.0,
            "min_delta_momentum": 1.20,
            "max_cur_cos": 0.10,
            "min_swap_cos": 0.60,
            "max_swaps_per_dataset": 50,
            "use_descriptors": False,
        }
        new_edges, audit = untangle_crossover_swaps(coords, edges, scale, None, xo_cfg)
        self.assertEqual(audit["crossovers_untangled"], 0)
        np.testing.assert_array_equal(new_edges, edges)

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
        self.assertIn((1, 2), set(map(tuple, retained_edges)))
        self.assertNotIn((1, 3), set(map(tuple, retained_edges)))

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
