"""Unit tests for Campaign 041 Triad Cytokinesis Invariants (TCI)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_041 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation041Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_041")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertTrue(v1["tci_enabled"])
        self.assertEqual(v1["max_daughter_separation_um"], 7.5)
        self.assertEqual(v1["max_daughter_separation_6bba_um"], 10.0)
        self.assertEqual(v1["min_mother_track_len"], 1)
        self.assertEqual(v1["max_birth_dist_um"], 5.5)
        self.assertEqual(v1["max_birth_dist_6bba_um"], 7.5)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertTrue(v2["tci_enabled"])
        self.assertEqual(v2["max_daughter_separation_um"], 8.0)
        self.assertEqual(v2["max_daughter_separation_6bba_um"], 11.0)
        self.assertEqual(v2["min_mother_track_len"], 1)
        self.assertEqual(v2["max_birth_dist_um"], 6.0)
        self.assertEqual(v2["max_birth_dist_6bba_um"], 8.0)

    def test_tci_protects_true_cytokinesis(self):
        """TCI protects true cytokinesis conforming to all 3 physical geometric laws."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        # Mother cell track: 0 -> 1 -> 2 (mother_len = 2 frames before dividing at 2)
        # At t=2, cell 2 divides into 3 (t=3) and 4 (t=3)
        # Daughter 1: moves +y (d1 = 1.625 um)
        # Daughter 2: moves -y (d2 = 1.625 um)
        # Daughter separation: d12 = 3.25 um (<= 7.5 um)
        # Mother length: 2 (>= 1)
        # Max birth dist: 1.625 um (<= 5.5 um)
        # Daughters persist for 5 frames each
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: grandmother
            [1.0, 10.0, 10.0, 10.0],  # 1: mother pre-division
            [2.0, 10.0, 10.0, 10.0],  # 2: mother dividing
            # Daughter 1 (persists 5 frames):
            [3.0, 10.0, 14.0, 10.0],  # 3 (dy = +4 = 1.625 um)
            [4.0, 10.0, 18.0, 10.0],  # 4
            [5.0, 10.0, 22.0, 10.0],  # 5
            [6.0, 10.0, 26.0, 10.0],  # 6
            [7.0, 10.0, 30.0, 10.0],  # 7
            # Daughter 2 (persists 5 frames):
            [3.0, 10.0,  6.0, 10.0],  # 8 (dy = -4 = -1.625 um)
            [4.0, 10.0,  2.0, 10.0],  # 9
            [5.0, 10.0, -2.0, 10.0],  # 10
            [6.0, 10.0, -6.0, 10.0],  # 11
            [7.0, 10.0, -10.0, 10.0], # 12
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
            [2, 8], [8, 9], [9, 10], [10, 11], [11, 12]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "divergent_pers_min": 3,
            "tilted_pers_min": 5,
            "tilted_max_birth_dist_um": 6.0,
            "tci_enabled": True,
            "max_daughter_separation_um": 7.5,
            "min_mother_track_len": 1,
            "max_birth_dist_um": 5.5,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)

    def test_tci_prunes_distant_daughter_separation(self):
        """TCI prunes forks where daughter separation exceeds physical cleavage limits (> 7.5 um)."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],  # 0: mother pre
            [1.0, 10.0, 10.0, 10.0],  # 1: mother
            # Daughter 1 (close, d1 = 1.625 um):
            [2.0, 10.0, 14.0, 10.0],  # 2
            [3.0, 10.0, 18.0, 10.0],  # 3
            # Daughter 2 (far, d2 = 8.125 um -> d_daughters = 9.75 um > 7.5 um):
            [2.0, 10.0, 10.0, 30.0],  # 4 (dx = +20 voxels = 8.125 um)
            [3.0, 10.0, 10.0, 34.0],  # 5
        ], dtype=np.float64)

        edges = np.array([
            [0, 1],
            [1, 2], [2, 3],
            [1, 4], [4, 5]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "divergent_pers_min": 3,
            "tilted_pers_min": 5,
            "tilted_max_birth_dist_um": 6.0,
            "tci_enabled": True,
            "max_daughter_separation_um": 7.5,
            "min_mother_track_len": 1,
            "max_birth_dist_um": 5.5,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        self.assertEqual(len(res_edges), len(edges) - 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertIn([1, 2], res_edges.tolist())
        self.assertNotIn([1, 4], res_edges.tolist())

    def test_tci_prunes_immediate_entry_division(self):
        """TCI prunes unphysical division at first detection mid-acquisition without mother track history."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            # Mother 0 appears out of nowhere at t=5 (in_degree == 0, mother_len == 0, mid-acquisition t > 0)
            [5.0, 10.0, 10.0, 10.0],  # 0: mother entry spur
            # Branch 1 (close, d1 = 0.81 um)
            [6.0, 10.0, 12.0, 10.0],  # 1
            [7.0, 10.0, 14.0, 10.0],  # 2
            # Branch 2 (d2 = 3.25 um, cos = -0.5)
            [6.0,  8.0, 10.0, 10.0],  # 3
            [7.0,  6.0, 10.0, 10.0],  # 4
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2],
            [0, 3], [3, 4]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "divergent_pers_min": 3,
            "tilted_pers_min": 5,
            "tilted_max_birth_dist_um": 6.0,
            "tci_enabled": True,
            "max_daughter_separation_um": 7.5,
            "min_mother_track_len": 1,
            "max_birth_dist_um": 5.5,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        self.assertEqual(len(res_edges), len(edges) - 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertIn([0, 1], res_edges.tolist())
        self.assertNotIn([0, 3], res_edges.tolist())

    def test_tci_protects_boundary_initial_frame_cytokinesis(self):
        """TCI protects genuine cytokinesis occurring at initial acquisition frame (t=0, mother_len=0)."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            # Mother at t=0 (pre-existing cell dividing at acquisition start)
            [0.0, 10.0, 10.0, 10.0],  # 0: mother
            # Daughter 1 (persists 4 frames)
            [1.0, 10.0, 14.0, 10.0],  # 1 (d1 = 1.625 um)
            [2.0, 10.0, 18.0, 10.0],  # 2
            [3.0, 10.0, 22.0, 10.0],  # 3
            [4.0, 10.0, 26.0, 10.0],  # 4
            # Daughter 2 (persists 4 frames, opposite direction cos = -1.0)
            [1.0, 10.0,  6.0, 10.0],  # 5 (d2 = 1.625 um)
            [2.0, 10.0,  2.0, 10.0],  # 6
            [3.0, 10.0, -2.0, 10.0],  # 7
            [4.0, 10.0, -6.0, 10.0],  # 8
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8]
        ], dtype=np.int64)

        prune_cfg = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "heading_guard": True,
            "heading_guard_cos_max": 0.20,
            "guard_requires_cos": True,
            "divergent_pers_min": 3,
            "tilted_pers_min": 5,
            "tilted_max_birth_dist_um": 6.0,
            "tci_enabled": True,
            "max_daughter_separation_um": 7.5,
            "min_mother_track_len": 1,
            "max_birth_dist_um": 5.5,
        }
        res_edges, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, prune_cfg)
        self.assertEqual(len(res_edges), len(edges))
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertIn([0, 1], res_edges.tolist())
        self.assertIn([0, 5], res_edges.tolist())

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
            "weight_xy": 0.0,
        }
        smoothed, audit = apply_harmonized_axial_regularizer(coords, edges, scale, sm_cfg)
        self.assertEqual(audit["spikes_suppressed"], 1)

    def test_build_candidate_graph_integrity(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 12.0, 10.0],
            [2.0, 10.0, 14.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        lineage = Lineage(coords, edges)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        cfg = config()
        v1_cfg = cfg["variants"]["variant_1"]
        cand, audit = build_candidate_graph(lineage, scale, None, None, v1_cfg, "44b6")
        self.assertEqual(len(cand.coords), len(coords))
        self.assertEqual(len(cand.edges), len(edges))


if __name__ == "__main__":
    unittest.main()
