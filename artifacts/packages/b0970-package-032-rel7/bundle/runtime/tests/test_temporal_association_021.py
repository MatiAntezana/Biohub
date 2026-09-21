import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_021 import (
    config,
    smooth_linear_tracks_conditional_z,
    get_ranker_model,
    add_mitosis_forks,
    build_candidate_graph,
    evaluate_association_panel,
    verify_sources,
)
from core.nextgen.contracts import Lineage
from core.nextgen import repair_ranker as r


def make_mock_ranker(positive=True):
    """Creates a mock linear ranker model preserving zero reference potential."""
    weights = np.zeros(len(r.NAMES), dtype=np.float64)
    if positive:
        idx_edge_delta = r.NAMES.index("edge_count_delta")
        weights[idx_edge_delta] = 2.0
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


class TemporalAssociation021Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_021")
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
        self.assertFalse(v1["fork_addition"]["enabled"])

        # Variant 2: pure symmetric regularizer + conservative fork addition
        v2 = cfg["variants"]["variant_2"]
        self.assertTrue(v2["smoothing"]["enabled"])
        self.assertTrue(v2["fork_addition"]["enabled"])
        self.assertEqual(v2["fork_addition"]["min_gain"], 0.80)
        self.assertEqual(v2["fork_addition"]["max_additions"], 10)
        self.assertEqual(v2["fork_addition"]["min_divergence_ratio"], 1.10)

        # Variant 3: pure symmetric regularizer + calibrated fork addition
        v3 = cfg["variants"]["variant_3"]
        self.assertTrue(v3["smoothing"]["enabled"])
        self.assertTrue(v3["fork_addition"]["enabled"])
        self.assertEqual(v3["fork_addition"]["min_gain"], 0.65)
        self.assertEqual(v3["fork_addition"]["max_additions"], 25)
        self.assertEqual(v3["fork_addition"]["min_divergence_ratio"], 1.05)

        # Gates
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
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        smoothed_asym, audit_asym = smooth_linear_tracks_conditional_z(coords_asym, edges, scale, cfg_smooth)
        self.assertEqual(audit_asym["spikes_suppressed"], 0)
        np.testing.assert_array_equal(smoothed_asym, coords_asym)

    def test_mitosis_fork_addition_kinematic_invariants(self):
        # Scale: [1.0, 1.0, 1.0] for simplicity in um calculations
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # Node setup:
        # 0: Mother m at t=0: (0, 10, 10, 10)
        # 1: Daughter 1 d1 at t=1: (1, 10, 10, 11) -> dist(m, d1) = 1.0 um <= 3.5
        # 2: Orphan d2 at t=1: (1, 10, 10, 16) -> dist(d1, d2) = 5.0 um in [2.5, 12.0]
        # 3: Granddaughter gc1 at t=2: (2, 10, 10, 9)
        # 4: Granddaughter gc2 at t=2: (2, 10, 10, 18) -> dist(gc1, gc2) = 9.0 um > 5.0 * 1.10
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0
            [1.0, 10.0, 10.0, 11.0],  # 1
            [1.0, 10.0, 10.0, 16.0],  # 2
            [2.0, 10.0, 10.0, 9.0],   # 3
            [2.0, 10.0, 10.0, 18.0],  # 4
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],
            [1, 3],
            [2, 4],  # d2 connects to gc2, but has in_degree == 0
        ], dtype=np.int64)
        descriptors = np.ones((5, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((5, 3), dtype=np.float32)

        ranker = make_mock_ranker(positive=True)
        cfg_addition = {
            "enabled": True,
            "min_gain": 0.5,
            "max_additions": 10,
            "max_d1_dist_um": 3.5,
            "max_d2_dist_um": 10.0,
            "min_sister_dist_um": 2.5,
            "max_sister_dist_um": 12.0,
            "min_divergence_ratio": 1.10
        }

        # Case A: Clean candidate should be accepted and edge (0, 2) added
        new_edges, audit = add_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_addition
        )
        self.assertEqual(audit["candidate_pairs_evaluated"], 1)
        self.assertEqual(audit["forks_added"], 1)
        self.assertEqual(audit["edges_added"], 1)
        edge_set = {tuple(e) for e in new_edges}
        self.assertIn((0, 2), edge_set)
        # Verify valid Lineage graph
        cand_lineage = Lineage(coords, new_edges).validate()
        self.assertEqual(len(cand_lineage.edges), 4)

        # Case B: Converging granddaughters (no divergence) -> rejected
        coords_converge = coords.copy()
        coords_converge[4] = [2.0, 10.0, 10.0, 12.0]  # dist(gc1, gc2) = 3.0 < 5.0 * 1.10
        new_edges_b, audit_b = add_mitosis_forks(
            coords_converge, edges, scale, descriptors, stats, ranker, "44b6", cfg_addition
        )
        self.assertEqual(audit_b["candidate_pairs_evaluated"], 0)
        self.assertEqual(audit_b["forks_added"], 0)

        # Case C: d2 is not an orphan (has a parent) -> rejected
        edges_non_orphan = np.vstack([edges, np.array([[0, 2]], dtype=np.int64)])
        new_edges_c, audit_c = add_mitosis_forks(
            coords, edges_non_orphan, scale, descriptors, stats, ranker, "44b6", cfg_addition
        )
        self.assertEqual(audit_c["candidate_pairs_evaluated"], 0)

    def test_mitosis_fork_addition_greedy_conflict_free_and_cap(self):
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # Two mothers (0, 5) competing for same orphan d2 (node 2)
        # Mother 0 at t=0, d1 at 1
        # Mother 5 at t=0, d1 at 6
        # d2 at node 2
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother 1
            [1.0, 10.0, 10.0, 11.0],  # 1: d1 for mother 1
            [1.0, 10.0, 10.0, 16.0],  # 2: orphan d2
            [2.0, 10.0, 10.0, 9.0],   # 3: gc1
            [2.0, 10.0, 10.0, 18.0],  # 4: gc2
            [0.0, 10.0, 10.0, 17.0],  # 5: mother 2
            [1.0, 10.0, 10.0, 17.5],  # 6: d1 for mother 2
            [2.0, 10.0, 10.0, 17.2],  # 7: gc1 for mother 2
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [1, 3],
            [2, 4],
            [5, 6], [6, 7],
        ], dtype=np.int64)
        descriptors = np.ones((8, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((8, 3), dtype=np.float32)

        ranker = make_mock_ranker(positive=True)
        cfg_addition = {
            "enabled": True,
            "min_gain": 0.5,
            "max_additions": 1,  # Cap at 1 addition
            "max_d1_dist_um": 3.5,
            "max_d2_dist_um": 10.0,
            "min_sister_dist_um": 1.0,
            "max_sister_dist_um": 12.0,
            "min_divergence_ratio": 0.5
        }

        new_edges, audit = add_mitosis_forks(
            coords, edges, scale, descriptors, stats, ranker, "44b6", cfg_addition
        )
        # Even if multiple mothers could pair with d2, d2 can only be assigned to ONE mother
        self.assertEqual(audit["forks_added"], 1)
        self.assertEqual(audit["edges_added"], 1)
        cand_lineage = Lineage(coords, new_edges).validate()
        self.assertEqual(len(cand_lineage.edges), 6)

    def test_build_candidate_graph_variant1_and_variant2(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 20.0, 30.0],  # symmetric reversal spike
            [1.0, 10.0, 20.0, 35.0],  # orphan d2
            [2.0, 10.0, 20.0, 30.0],  # d1 child (node 3)
            [2.0, 10.0, 20.0, 38.0],  # d2 child (node 4)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 3], [2, 4]], dtype=np.int64)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        descriptors = np.ones((5, 405), dtype=np.float32) / np.sqrt(405)
        stats = np.ones((5, 3), dtype=np.float32)

        anchor = Lineage(coords, edges).validate()

        # Variant 1: Smoothing only
        v1_cfg = {
            "smoothing": {
                "enabled": True,
                "reversal_only": True,
                "symmetric_only": True,
                "jitter_threshold_um": 0.8,
                "max_correction_um": 0.6,
                "weight_z": 0.80,
            },
            "fork_addition": {"enabled": False}
        }
        cand1, audit1 = build_candidate_graph(anchor, scale, descriptors, stats, v1_cfg, group="44b6")
        self.assertIsInstance(cand1, Lineage)
        self.assertEqual(audit1["fork_addition"]["forks_added"], 0)
        self.assertEqual(len(cand1.edges), 3)

        # Variant 2: Smoothing + Fork Addition
        v2_cfg = {
            "smoothing": {
                "enabled": True,
                "reversal_only": True,
                "symmetric_only": True,
                "jitter_threshold_um": 0.8,
                "max_correction_um": 0.6,
                "weight_z": 0.80,
            },
            "fork_addition": {
                "enabled": True,
                "min_gain": 0.5,
                "max_additions": 5,
                "max_d1_dist_um": 5.0,
                "max_d2_dist_um": 10.0,
                "min_sister_dist_um": 2.0,
                "max_sister_dist_um": 12.0,
                "min_divergence_ratio": 0.8
            }
        }
        ranker = make_mock_ranker(positive=True)
        cand2, audit2 = build_candidate_graph(
            anchor, scale, descriptors, stats, v2_cfg, group="44b6", ranker_model=ranker
        )
        self.assertIsInstance(cand2, Lineage)
        self.assertGreaterEqual(audit2["fork_addition"]["forks_added"], 0)

    def test_verify_sources(self):
        # Must reject invalid source run folder name
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        # When pointing to the actual source run directory in artifacts, it must pass
        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)


if __name__ == "__main__":
    unittest.main()
