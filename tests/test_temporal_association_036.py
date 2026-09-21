"""Unit tests for Campaign 036 Drift-Compensated Linear Track Extension (DC-LTE) & Bio-Conserved Mitosis Gating (BC-MG)."""
from collections import Counter
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_036 import (
    config,
    verify_sources,
    compute_global_drift,
    apply_dc_lte,
    apply_bc_mg,
    build_candidate_graph,
)


class TemporalAssociation036Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_036")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])
        self.assertIn("gates", cfg)

        for v_name in ("variant_1", "variant_2", "variant_3"):
            v = cfg["variants"][v_name]
            self.assertIn("smoothing", v)
            self.assertIn("fork_pruning", v)
            # self.assertIn("dc_lte", v)
            self.assertIn("mitosis_reassignment", v)

            self.assertIn("cardinality_optimization", v)
            self.assertIn("mitosis_reassignment", v)
            card = v["cardinality_optimization"]
            mr = v["mitosis_reassignment"]
            if mr.get("enabled", False):
                self.assertLessEqual(mr["cos_max"], -0.50)
                self.assertGreaterEqual(mr["mother_history_min_frames"], 3)
                self.assertGreaterEqual(mr["daughter_persistence_min_frames"], 2)

    def test_compute_global_drift(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        disp_um = np.array([0.5, 1.0, -0.5], dtype=np.float64)
        disp_pix = disp_um / scale

        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [0.0, 15.0, 20.0, 15.0],
            [0.0, 20.0, 15.0, 25.0],
            [1.0, 10.0 + disp_pix[0], 10.0 + disp_pix[1], 10.0 + disp_pix[2]],
            [1.0, 15.0 + disp_pix[0], 20.0 + disp_pix[1], 15.0 + disp_pix[2]],
            [1.0, 20.0 + disp_pix[0], 15.0 + disp_pix[1], 25.0 + disp_pix[2]],
        ], dtype=np.float64)
        edges = np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int64)

        drifts = compute_global_drift(coords, edges, scale)
        self.assertIn(0, drifts)
        np.testing.assert_allclose(drifts[0], disp_um, atol=1e-5)

    def test_dc_lte_extension_basic(self):
        # Node 0 at t=0 terminates (out_degree == 0)
        # Node 1 at t=1 is orphan (in_degree == 0), located 1.0 um away
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0 + (1.0 / scale[1]), 10.0],
        ], dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)

        lte_cfg = {
            "enabled": True,
            "d_max_um": 2.0,
            "p_candidate_min": 0.0,
            "sim_min": 0.0,
            "global_drift_compensation": False,
        }

        new_edges, audit = apply_dc_lte(coords, edges, scale, lte_cfg)
        self.assertEqual(audit["extensions"], 1)
        self.assertEqual(len(new_edges), 1)
        self.assertEqual(tuple(new_edges[0]), (0, 1))

        # Graph invariants: strictly out_degree <= 1, in_degree <= 1, zero divisions
        lineage = Lineage(coords, new_edges)
        lineage.validate()
        from collections import Counter
        div_count = sum(1 for cnt in Counter(lineage.edges[:, 0]).values() if cnt > 1)
        self.assertEqual(div_count, 0)

    def test_dc_lte_respects_d_max(self):
        # Node 0 at t=0, Node 1 at t=1 is 3.5 um away (exceeds d_max_um=2.0)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0 + (3.5 / scale[1]), 10.0],
        ], dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)

        lte_cfg = {
            "enabled": True,
            "d_max_um": 2.0,
            "p_candidate_min": 0.0,
            "sim_min": 0.0,
            "global_drift_compensation": False,
        }

        new_edges, audit = apply_dc_lte(coords, edges, scale, lte_cfg)
        self.assertEqual(audit["extensions"], 0)
        self.assertEqual(len(new_edges), 0)

    def test_dc_lte_disjoint_1_to_1_matching(self):
        # 1 terminal node at t=0 (node 0)
        # 2 orphan nodes at t=1: node 1 (1.0 um away), node 2 (1.5 um away)
        # Greedy matching must pick node 1 only; out_degree of node 0 must be 1
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],                     # 0
            [1.0, 10.0, 10.0 + (1.0 / scale[1]), 10.0],  # 1 (closer)
            [1.0, 10.0, 10.0 + (1.5 / scale[1]), 10.0],  # 2 (farther)
        ], dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)

        lte_cfg = {
            "enabled": True,
            "d_max_um": 2.0,
            "p_candidate_min": 0.0,
            "sim_min": 0.0,
            "global_drift_compensation": False,
        }

        new_edges, audit = apply_dc_lte(coords, edges, scale, lte_cfg)
        self.assertEqual(audit["extensions"], 1)
        self.assertEqual(tuple(new_edges[0]), (0, 1))

        # Check lineage
        lineage = Lineage(coords, new_edges)
        lineage.validate()
        div_count = sum(1 for cnt in Counter(lineage.edges[:, 0]).values() if cnt > 1)
        self.assertEqual(div_count, 0)

    def test_bc_mg_rejects_insufficient_history(self):
        # Mother m at t=1 has child d1 at t=2
        # Track history of m is only 1 node (started at t=1, no parent)
        # 6-layer gate requires history >= 3, so reassignment must be rejected
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [1.0, 10.0, 20.0, 20.0],  # m (0) at t=1
            [1.0, 10.0, 15.0, 20.0],  # c (1) at t=1
            [2.0, 10.0, 25.0, 20.0],  # d1 (2) at t=2
            [2.0, 10.0, 15.0, 20.0],  # d2 (3) at t=2
        ], dtype=np.float64)
        edges = np.array([[0, 2], [1, 3]], dtype=np.int64)

        reassign_cfg = {
            "enabled": True,
            "method": "bc_mg",
            "global_drift_compensation": False,
            "cos_max": -0.50,
            "sister_distance_min_um": 2.0,
            "sister_distance_max_um": 16.0,
            "distance_ratio_min": 0.15,
            "p_candidate_min": 0.0,
            "mother_history_min_frames": 3,
            "daughter_persistence_min_frames": 1,
        }

        new_edges, audit = apply_bc_mg(coords, edges, scale, reassign_cfg)
        self.assertEqual(audit["reassignments"], 0)
        self.assertEqual(len(new_edges), len(edges))

    def test_build_candidate_graph_pipeline(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.0 + (1.0 / scale[1]), 10.0],
            [2.0, 10.0, 10.0 + (2.0 / scale[1]), 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1]], dtype=np.int64)
        anchor = Lineage(coords, edges)

        cfg = config()["variants"]["variant_1"]
        cand, audit = build_candidate_graph(anchor, scale, None, None, cfg, group="test")
        cand.validate()
        self.assertIn("smoothing", audit)
        self.assertIn("fork_pruning", audit)
        self.assertIn("mitosis_reassignment", audit)
        self.assertIn("smoothing", audit)
        self.assertEqual(len(cand.edges), 1)
        div_count = sum(1 for cnt in Counter(cand.edges[:, 0]).values() if cnt > 1)
        self.assertEqual(div_count, 0)


if __name__ == "__main__":
    unittest.main()
