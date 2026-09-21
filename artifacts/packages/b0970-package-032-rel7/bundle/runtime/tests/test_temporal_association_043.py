"""Unit tests for Campaign 043 Adaptive Allometric Cytokinesis Gating (AACG)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_043 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    apply_de_novo_daughter_pairing,
)


class TemporalAssociation043Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_043")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["aacg_enabled"])
        self.assertEqual(v1["max_daughter_separation_44b6_um"], 7.2)
        self.assertEqual(v1["max_daughter_separation_6bba_um"], 13.5)
        self.assertEqual(v1["min_mother_track_len"], 1)
        self.assertEqual(v1["max_birth_dist_44b6_um"], 6.0)
        self.assertEqual(v1["max_birth_dist_6bba_um"], 7.5)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["aacg_enabled"])
        self.assertEqual(v2["max_daughter_separation_44b6_um"], 7.5)
        self.assertEqual(v2["max_daughter_separation_6bba_um"], 14.5)
        self.assertEqual(v2["min_mother_track_len"], 1)
        self.assertEqual(v2["max_birth_dist_44b6_um"], 6.2)
        self.assertEqual(v2["max_birth_dist_6bba_um"], 8.0)

    def test_aacg_protects_true_cytokinesis_44b6(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0, 10.0],
            [2.0, 10.0, 10.0, 10.0],
            # Daughter 1 (dy = +4 = 1.625 um):
            [3.0, 10.0, 14.0, 10.0],
            [4.0, 10.0, 18.0, 10.0],
            [5.0, 10.0, 22.0, 10.0],
            [6.0, 10.0, 26.0, 10.0],
            [7.0, 10.0, 30.0, 10.0],
            # Daughter 2 (dy = -4 = -1.625 um):
            [3.0, 10.0,  6.0, 10.0],
            [4.0, 10.0,  2.0, 10.0],
            [5.0, 10.0, -2.0, 10.0],
            [6.0, 10.0, -6.0, 10.0],
            [7.0, 10.0, -10.0, 10.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [2, 8], [8, 9], [9, 10], [10, 11], [11, 12]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="44b6")
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)

    def test_aacg_protects_true_cytokinesis_6bba(self):
        """In 6bba, fast-motility daughters separate up to 11.5 um and must be protected."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother divides at t=2
        # Daughter 1 moves dy = +14 = 5.6875 um
        # Daughter 2 moves dy = -14 = -5.6875 um
        # Daughter separation = 11.375 um (well within 13.5 um 6bba limit)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 10.0, 20.0, 20.0],
            [2.0, 10.0, 20.0, 20.0],
            # Daughter 1:
            [3.0, 10.0, 34.0, 20.0],
            [4.0, 10.0, 38.0, 20.0],
            [5.0, 10.0, 42.0, 20.0],
            # Daughter 2:
            [3.0, 10.0,  6.0, 20.0],
            [4.0, 10.0,  2.0, 20.0],
            [5.0, 10.0, -2.0, 20.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5],
            [2, 6], [6, 7], [7, 8]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="6bba")
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)

    def test_aacg_prunes_excessive_separation_44b6(self):
        """In 44b6, daughter separation > 7.2 um is unphysical and gets pruned."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Daughter separation = 8.9375 um > 7.2 um
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [1.0, 10.0, 20.0, 20.0],
            [2.0, 10.0, 20.0, 20.0],
            # Daughter 1: dy = +11 = 4.47 um
            [3.0, 10.0, 31.0, 20.0],
            [4.0, 10.0, 35.0, 20.0],
            [5.0, 10.0, 39.0, 20.0],
            # Daughter 2: dy = -11 = -4.47 um -> sep = 8.94 um
            [3.0, 10.0,  9.0, 20.0],
            [4.0, 10.0,  5.0, 20.0],
            [5.0, 10.0,  1.0, 20.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5],
            [2, 6], [6, 7], [7, 8]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="44b6")
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(res_edges), len(edges) - 1)

    def test_aacg_initial_frame_mother_boundary(self):
        """Cells dividing at t=0.0 with mlen=0 must not be pruned by maturation check."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # mother dividing at initial frame t=0.0
            [1.0, 10.0, 14.0, 10.0],  # daughter 1
            [2.0, 10.0, 18.0, 10.0],
            [3.0, 10.0, 22.0, 10.0],
            [1.0, 10.0,  6.0, 10.0],  # daughter 2
            [2.0, 10.0,  2.0, 10.0],
            [3.0, 10.0, -2.0, 10.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3],
            [0, 4], [4, 5], [5, 6]
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["fork_pruning"]
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, cfg, group="44b6")
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)




    def test_dndp_recovers_missing_divergent_daughter_44b6(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother at t=0, (0, 10, 10, 10)
        # D1 at t=1, (1, 10, 14, 10) -> dy = +4 vox = +1.625 um
        # D2 at t=1, (1, 10, 6, 10)  -> dy = -4 vox = -1.625 um (antiparallel: cos = -1.0)
        # D1 is linked (0 -> 1)
        # D2 is orphan start (2 -> 3 -> 4, length 2 >= orphan_pers_min)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 14.0, 10.0],
            [1.0, 10.0,  6.0, 10.0],
            [2.0, 10.0,  5.0, 10.0],
            [3.0, 10.0,  4.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],
            [2, 3],
            [3, 4],
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["dndp"]
        new_edges, audit = apply_de_novo_daughter_pairing(coords, edges, scale, cfg, group="44b6")
        self.assertEqual(audit["pairings_added"], 1)
        self.assertIn([0, 2], new_edges.tolist())

    def test_dndp_recovers_missing_divergent_daughter_6bba(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # In 6bba: daughters separate further (e.g. sep = 11.375 um)
        # Mother at t=1: (1, 10, 10, 10), predecessor at t=0 (0, 10, 10, 10)
        # D1 at t=2: (2, 10, 24, 10) -> dy = +14 = +5.6875 um
        # D2 at t=2: (2, 10, -4, 10) -> dy = -14 = -5.6875 um (sep = 11.375 um <= 15.0 um)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0, 10.0],
            [2.0, 10.0, 24.0, 10.0],
            [2.0, 10.0, -4.0, 10.0],
            [3.0, 10.0, -5.0, 10.0],
            [4.0, 10.0, -6.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],
            [1, 2], # M -> D1
            [3, 4], # D2 continuation
            [4, 5],
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["dndp"]
        new_edges, audit = apply_de_novo_daughter_pairing(coords, edges, scale, cfg, group="6bba")
        self.assertEqual(audit["pairings_added"], 1)
        self.assertIn([1, 3], new_edges.tolist())

    def test_dndp_rejects_parallel_crossover_daughter(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # D1 and candidate orphan D2 both move in positive Y direction: cos = +1.0 (parallel)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 14.0, 10.0], # D1: dy = +4
            [1.0, 10.0, 15.0, 10.0], # D2: dy = +5 (cos ~ +1.0, NOT cytokinesis)
            [2.0, 10.0, 16.0, 10.0],
            [3.0, 10.0, 17.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],
            [2, 3],
            [3, 4],
        ], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["dndp"]
        new_edges, audit = apply_de_novo_daughter_pairing(coords, edges, scale, cfg, group="44b6")
        self.assertEqual(audit["pairings_added"], 0)
        self.assertEqual(len(new_edges), len(edges))

    def test_dndp_rejects_ephemeral_noise_orphan(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # D2 has no subsequent detections (track_len = 0 < orphan_pers_min = 2)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 14.0, 10.0],
            [1.0, 10.0,  6.0, 10.0], # D2 is a single-frame noise blip
        ], dtype=np.float64)
        edges = np.array([[0, 1]], dtype=np.int64)

        cfg = config()["variants"]["variant_1"]["dndp"]
        new_edges, audit = apply_de_novo_daughter_pairing(coords, edges, scale, cfg, group="44b6")
        self.assertEqual(audit["pairings_added"], 0)
        self.assertEqual(len(new_edges), len(edges))

    def test_build_candidate_graph_integrity_043(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 14.0, 10.0],
            [1.0, 10.0,  6.0, 10.0],
            [2.0, 10.0,  5.0, 10.0],
            [3.0, 10.0,  4.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.int64)
        anchor = Lineage(coords, edges)
        v_cfg = config()["variants"]["variant_1"]

        cand, audit = build_candidate_graph(anchor, scale, None, {}, v_cfg, group="44b6")
        self.assertIsInstance(cand, Lineage)
        self.assertIn("dndp", audit)
        self.assertEqual(audit["dndp"]["pairings_added"], 1)
        # Ensure graph validity (in_degree <= 1, out_degree <= 2)
        cand.validate()

if __name__ == "__main__":
    unittest.main()
