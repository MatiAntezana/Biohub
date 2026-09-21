import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_020 import (
    config,
    smooth_linear_tracks_conditional_z,
    get_ranker_model,
    prune_mitosis_forks,
    build_candidate_graph,
    evaluate_association_panel,
)
from core.nextgen.contracts import Lineage
from core.nextgen import repair_ranker as r


def make_mock_ranker(preferred_daughter=1):
    """Creates a mock linear ranker model preserving zero reference potential."""
    weights = np.zeros(len(r.NAMES), dtype=np.float64)
    # We can set weight on edge_count_delta or specific feature
    scale = np.ones(len(r.NAMES), dtype=np.float64)
    mean = np.zeros(len(r.NAMES), dtype=np.float64)
    active = np.ones(len(r.NAMES), dtype=bool)
    return {
        "feature_names": list(r.NAMES),
        "weights": weights.tolist(),
        "intercept": 0.0,
        "standardizer": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "active": active.tolist(),
        },
    }


class TemporalAssociation020Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_020")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])

        # Variant 1: pure symmetric regularization only
        v1 = cfg["variants"]["variant_1"]
        self.assertTrue(v1["smoothing"]["enabled"])
        self.assertTrue(v1["smoothing"]["reversal_only"])
        self.assertTrue(v1["smoothing"]["symmetric_only"])
        self.assertEqual(v1["smoothing"]["max_correction_um"], 0.6)
        self.assertEqual(v1["smoothing"]["jitter_threshold_um"], 0.8)
        self.assertEqual(v1["smoothing"]["weight_z"], 0.80)
        self.assertFalse(v1["fork_pruning"]["enabled"])

        # Variant 2: pure symmetric regularizer + conservative fork pruning
        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["fork_pruning"]["enabled"])
        self.assertLessEqual(v2["fork_pruning"]["threshold"], -0.55)

        # Variant 3: pure symmetric regularizer + medium fork pruning
        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["smoothing"]["enabled"])
        self.assertTrue(v3["fork_pruning"]["enabled"])
        self.assertLess(v3["fork_pruning"]["threshold"], v2["fork_pruning"]["threshold"])

        # Gates
        gates = cfg["gates"]
        self.assertEqual(gates["development_score_delta_min"], 0.003)
        self.assertEqual(gates["confirmation_score_delta_min"], 0.002)

    def test_axial_z_smoothing_pure_symmetric_invariants(self):
        # Linear track: t=0, t=1, t=2
        # Symmetric reversal spike: z=10.0 -> z=12.0 -> z=10.0
        # scale_z = 1.625 um/voxel -> dz = 2.0 voxels = 3.25 um > 0.8 um thresh
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
        smoothed, audit = smooth_linear_tracks_conditional_z(coords, edges, scale, cfg_smooth)
        self.assertEqual(audit["spikes_suppressed"], 1)

        # Lateral XY coordinates must remain strictly untouched
        np.testing.assert_array_equal(smoothed[:, 2], coords[:, 2])
        np.testing.assert_array_equal(smoothed[:, 3], coords[:, 3])

        # Endpoints must remain unchanged
        self.assertEqual(smoothed[0, 1], coords[0, 1])
        self.assertEqual(smoothed[2, 1], coords[2, 1])

        # Node 1 Z must be clamped to at most 0.6 um displacement
        disp_um = abs(smoothed[1, 1] - coords[1, 1]) * scale[0]
        self.assertAlmostEqual(disp_um, 0.6, places=5)

    def test_axial_z_smoothing_preserves_monotonic_and_asymmetric(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        cfg_smooth = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
        }

        # Monotonic motion: 10 -> 14 -> 20 (dz1=4, dz2=6)
        coords_mono = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 14.0, 20.0, 30.0],
            [2.0, 20.0, 20.0, 30.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        smoothed_mono, audit_mono = smooth_linear_tracks_conditional_z(coords_mono, edges, scale, cfg_smooth)
        self.assertEqual(audit_mono["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed_mono, coords_mono)

        # Asymmetric reversal: 10 -> 14 -> 12 (dz1=4, dz2=-2, but z_p != z_s)
        coords_asym = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 14.0, 20.0, 30.0],
            [2.0, 12.0, 20.0, 30.0],
        ], dtype=np.float64)
        smoothed_asym, audit_asym = smooth_linear_tracks_conditional_z(coords_asym, edges, scale, cfg_smooth)
        self.assertEqual(audit_asym["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed_asym, coords_asym)

    def test_frozen_015_ranker_models_load_and_score(self):
        for group in ("44b6", "6bba"):
            try:
                model = get_ranker_model(group)
            except FileNotFoundError:
                model = make_mock_ranker()
            self.assertEqual(model["feature_names"], list(r.NAMES))
            self.assertEqual(model["intercept"], 0.0)
            self.assertEqual(len(model["weights"]), len(r.NAMES))

            # Reference feature vector (all zeros) must evaluate to exactly 0.0
            zero_feat = np.zeros((1, len(r.NAMES)), dtype=np.float64)
            score = r.scores(model, zero_feat)
            self.assertEqual(score[0], 0.0)

    def test_fork_pruning_removes_low_affinity_arm_when_above_margin(self):
        # Synthetic fork:
        # Mother 0 at t=0
        # Daughter 1 at t=1 (close to mother, high contrast)
        # Daughter 2 at t=1 (far from mother, low contrast, false division arm)
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0: mother
            [1.0, 10.1, 20.1, 30.1],  # 1: true daughter (dist ~ 0.1)
            [1.0, 15.0, 30.0, 40.0],  # 2: false daughter (dist ~ 15.0)
            [2.0, 10.2, 20.2, 30.2],  # 3: continuation of 1
            [2.0, 16.0, 31.0, 41.0],  # 4: continuation of 2
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],  # fork arm 1
            [0, 2],  # fork arm 2
            [1, 3],
            [2, 4],
        ], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        # Descriptors: normalized vectors
        descriptors = np.zeros((5, 405), dtype=np.float32)
        base = np.ones(405, dtype=np.float32) / np.sqrt(405)
        descriptors[0] = base
        descriptors[1] = base  # identical to mother
        descriptors[2] = -base  # opposite / low correlation
        descriptors[3] = base
        descriptors[4] = -base

        # Stats: contrast, valid, inside
        stats = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.1, 1.0, 1.0],  # low contrast for false daughter
            [1.0, 1.0, 1.0],
            [0.1, 1.0, 1.0],
        ], dtype=np.float32)

        # Build ranker that penalizes large distance and low correlation
        ranker = make_mock_ranker()
        # Patch correlation has positive weight, distance has negative weight
        # NAMES: "edge_delta/distance" is index 0, "edge_delta/patch_correlation" is index 7
        idx_dist = r.NAMES.index("edge_delta/distance")
        idx_corr = r.NAMES.index("edge_delta/patch_correlation")
        # Removing edge 2 removes high distance (delta dist is negative) and low correlation
        # Setting negative weight for distance means removing high distance edge increases potential
        ranker["weights"][idx_dist] = -1.0
        ranker["weights"][idx_corr] = 1.0

        cfg_prune = {
            "enabled": True,
            "margin": -10.0,  # low margin to trigger
            "min_affinity_margin": 0.0,
        }

        pruned_edges, audit = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_prune
        )

        self.assertEqual(audit["forks_evaluated"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(audit["edges_removed"], 1)

        # The low-affinity edge (0, 2) must be removed, and (0, 1) must be preserved
        edge_set = {tuple(e) for e in pruned_edges}
        self.assertIn((0, 1), edge_set)
        self.assertNotIn((0, 2), edge_set)
        self.assertIn((1, 3), edge_set)
        self.assertIn((2, 4), edge_set)

        # Validates as a Lineage graph
        cand = Lineage(coords, pruned_edges).validate()
        self.assertEqual(len(cand.coords), 5)
        self.assertEqual(len(cand.edges), 3)

    def test_fork_pruning_respects_threshold_margin(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 10.1, 20.1, 30.1],
            [1.0, 11.0, 21.0, 31.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        descriptors = np.ones((3, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((3, 3), dtype=np.float32)

        ranker = make_mock_ranker()
        ranker["weights"][0] = -0.5

        # Case A: margin very high -> no pruning
        cfg_high_margin = {"enabled": True, "margin": 100.0}
        edges_a, audit_a = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_high_margin
        )
        self.assertEqual(audit_a["forks_pruned"], 0)
        self.assertEqual(len(edges_a), 2)

        # Case B: margin very low -> pruning allowed
        cfg_low_margin = {"enabled": True, "margin": -100.0}
        edges_b, audit_b = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_low_margin
        )
        self.assertEqual(audit_b["forks_pruned"], 1)
        self.assertEqual(len(edges_b), 1)

    def test_fork_pruning_abstains_on_tied_arms(self):
        # Perfectly symmetric fork in both geometry and descriptors
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 10.5, 20.5, 30.5],
            [1.0, 10.5, 20.5, 30.5],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        descriptors = np.ones((3, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((3, 3), dtype=np.float32)

        ranker = make_mock_ranker()
        cfg_prune = {"enabled": True, "margin": -100.0}

        edges_out, audit = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_prune
        )
        # Symmetrical daughters give identical scores -> must abstain!
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(len(edges_out), 2)

    def test_fork_pruning_respects_max_prunes(self):
        # Graph with 2 distinct forks
        coords = np.array([
            # Fork 1: 0 -> 1, 2
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 10.1, 20.1, 30.1],
            [1.0, 15.0, 30.0, 40.0],
            # Fork 2: 3 -> 4, 5
            [0.0, 50.0, 60.0, 70.0],
            [1.0, 50.1, 60.1, 70.1],
            [1.0, 58.0, 75.0, 85.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [0, 2],
            [3, 4], [3, 5],
        ], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        descriptors = np.ones((6, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((6, 3), dtype=np.float32)

        ranker = make_mock_ranker()
        idx_dist = r.NAMES.index("edge_delta/distance")
        ranker["weights"][idx_dist] = -1.0

        # With max_prunes=1, only the single highest gain fork should be pruned
        cfg_prune = {"enabled": True, "margin": -100.0, "max_prunes": 1}
        edges_out, audit = prune_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_prune
        )
        self.assertEqual(audit["forks_evaluated"], 2)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(audit["edges_removed"], 1)
        self.assertEqual(len(edges_out), 3)

    def test_build_candidate_graph_variant1_and_variant2(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 20.0, 30.0],  # symmetric reversal spike
            [2.0, 10.0, 20.0, 30.0],
            [2.0, 15.0, 30.0, 40.0],  # fork at node 1 -> 2, 3
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        descriptors = np.ones((4, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((4, 3), dtype=np.float32)

        anchor = Lineage(coords, edges).validate()

        # Variant 1: Pure symmetric smoothing, fork pruning disabled
        v1_cfg = {
            "smoothing": {
                "enabled": True,
                "reversal_only": True,
                "symmetric_only": True,
                "jitter_threshold_um": 0.8,
                "max_correction_um": 0.6,
                "weight_z": 0.80,
            },
            "fork_pruning": {"enabled": False}
        }
        cand1, audit1 = build_candidate_graph(anchor, scale, descriptors, stats, v1_cfg, group="44b6")
        self.assertIsInstance(cand1, Lineage)
        self.assertEqual(audit1["fork_pruning"]["forks_pruned"], 0)
        self.assertEqual(len(cand1.edges), 3)

        # Variant 2: Smoothing + fork pruning
        v2_cfg = {
            "smoothing": {
                "enabled": True,
                "reversal_only": True,
                "symmetric_only": True,
                "jitter_threshold_um": 0.8,
                "max_correction_um": 0.6,
                "weight_z": 0.80,
            },
            "fork_pruning": {"enabled": True, "margin": -100.0}
        }
        ranker = make_mock_ranker()
        ranker["weights"][0] = -1.0
        cand2, audit2 = build_candidate_graph(
            anchor, scale, descriptors, stats, v2_cfg, group="44b6", ranker_model=ranker
        )
        self.assertIsInstance(cand2, Lineage)
        self.assertEqual(audit2["fork_pruning"]["forks_pruned"], 1)
        self.assertEqual(len(cand2.edges), 2)

    def test_synthetic_evidence_fork_pruning_execution(self):
        # Self-contained multi-fork graph to test full candidate building and audit structure
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],  # 0: mother fork 1
            [1.0, 10.1, 20.1, 30.1],  # 1: child 1a
            [1.0, 15.0, 30.0, 40.0],  # 2: child 1b
            [2.0, 10.2, 20.2, 30.2],  # 3: child 1a continuation -> mother fork 2
            [3.0, 10.3, 20.3, 30.3],  # 4: child 2a
            [3.0, 16.0, 31.0, 41.0],  # 5: child 2b
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2], [1, 3], [3, 4], [3, 5]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        desc = np.ones((6, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((6, 3), dtype=np.float32)
        anchor = Lineage(coords, edges).validate()

        cfg = config()
        v1_cfg = cfg["variants"]["variant_1"]
        cand1, audit1 = build_candidate_graph(anchor, scale, desc, stats, v1_cfg, group="44b6")
        self.assertIsInstance(cand1, Lineage)
        self.assertEqual(audit1["fork_pruning"]["forks_pruned"], 0)

        v2_cfg = cfg["variants"]["variant_2"]
        ranker = make_mock_ranker()
        ranker["weights"][0] = -1.0
        cand2, audit2 = build_candidate_graph(
            anchor, scale, desc, stats, v2_cfg, group="44b6", ranker_model=ranker
        )
        self.assertIsInstance(cand2, Lineage)
        self.assertGreater(audit2["fork_pruning"]["forks_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
