"""Unit tests for Campaign 033 Drift-Compensated Cytokinesis-directed Mitosis Reassignment (DC-CMR)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_033 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    compute_global_drift,
    apply_dc_cmr,
    build_candidate_graph,
)


class TemporalAssociation033Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_033")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])
        self.assertIn("gates", cfg)

        for v_name in ("variant_1", "variant_2", "variant_3"):
            v = cfg["variants"][v_name]
            self.assertIn("smoothing", v)
            self.assertIn("fork_pruning", v)
            self.assertIn("mitosis_reassignment", v)

            mr = v["mitosis_reassignment"]
            self.assertTrue(mr["enabled"])
            self.assertEqual(mr["method"], "dc_cmr")
            self.assertTrue(mr["global_drift_compensation"])
            self.assertLessEqual(mr["cos_max"], -0.25)
            self.assertGreaterEqual(mr["sister_distance_min_um"], 2.0)
            self.assertLessEqual(mr["sister_distance_max_um"], 20.0)
            self.assertGreaterEqual(mr["distance_ratio_min"], 0.10)
            self.assertFalse(mr.get("gap_interpolation_enabled", False))

    def test_compute_global_drift(self):
        # 4 cells at t=0 moving by [+0.5, +1.0, -0.5] um to t=1
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

    def test_dc_cmr_reassignment_basic(self):
        # Mother m (node 0) at t=0 has daughter d1 (node 2) at t=1 along +y axis (+2.0 um)
        # Competitor c (node 1) at t=0 has daughter d2 (node 3) at t=1 along -y axis (-2.0 um from m)
        # (m -> d1) and (m -> d2) form a perfect antiparallel cytokinesis pair (cos = -1.0)
        # Competitor c is a single-daughter track
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],   # m (0) at t=0
            [0.0, 10.0, 15.0, 20.0],   # c (1) at t=0 (nearby competitor)
            [1.0, 10.0, 25.0, 20.0],   # d1 (2) at t=1 (+5 pix in y = +2.03 um from m)
            [1.0, 10.0, 15.0, 20.0],   # d2 (3) at t=1 (-5 pix in y = -2.03 um from m)
        ], dtype=np.float64)
        edges = np.array([
            [0, 2],  # m -> d1
            [1, 3],  # c -> d2 (stolen daughter)
        ], dtype=np.int64)

        reassign_cfg = {
            "enabled": True,
            "method": "dc_cmr",
            "global_drift_compensation": False,
            "cos_max": -0.30,
            "sister_distance_min_um": 2.0,
            "sister_distance_max_um": 16.0,
            "distance_ratio_min": 0.15,
            "p_candidate_min": 0.0,  # no cand_npz supplied
            "competitor_max_out_deg": 1,
        }

        new_edges, audit = apply_dc_cmr(coords, edges, scale, reassign_cfg, cand_npz=None)
        self.assertEqual(audit["reassignments"], 1)
        new_edge_set = set(map(tuple, new_edges))

        # d2 must be reassigned from c to m!
        self.assertIn((0, 3), new_edge_set)  # m -> d2
        self.assertIn((0, 2), new_edge_set)  # m -> d1 retained
        self.assertNotIn((1, 3), new_edge_set)  # c -> d2 removed

    def test_dc_cmr_no_reassign_high_cos(self):
        # Two daughters moving in the SAME direction (parallel, cos = +1.0)
        # Must NOT be reassigned as cytokinesis!
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],   # m (0) at t=0
            [0.0, 10.0, 15.0, 20.0],   # c (1) at t=0
            [1.0, 10.0, 25.0, 20.0],   # d1 (2) at t=1 (+5 pix y)
            [1.0, 10.0, 26.0, 20.0],   # d2 (3) at t=1 (+6 pix y, parallel!)
        ], dtype=np.float64)
        edges = np.array([[0, 2], [1, 3]], dtype=np.int64)

        reassign_cfg = {
            "enabled": True,
            "cos_max": -0.30,
            "sister_distance_min_um": 0.1,
            "sister_distance_max_um": 16.0,
            "distance_ratio_min": 0.15,
            "p_candidate_min": 0.0,
            "competitor_max_out_deg": 1,
        }

        new_edges, audit = apply_dc_cmr(coords, edges, scale, reassign_cfg, cand_npz=None)
        self.assertEqual(audit["reassignments"], 0)
        new_edge_set = set(map(tuple, new_edges))
        self.assertIn((1, 3), new_edge_set)
        self.assertNotIn((0, 3), new_edge_set)

    def test_dc_cmr_preserves_lineage_invariants(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [0.0, 10.0, 15.0, 20.0],
            [1.0, 10.0, 25.0, 20.0],
            [1.0, 10.0, 15.0, 20.0],
        ], dtype=np.float64)
        edges = np.array([[0, 2], [1, 3]], dtype=np.int64)

        reassign_cfg = {
            "enabled": True,
            "cos_max": -0.30,
            "sister_distance_min_um": 2.0,
            "sister_distance_max_um": 16.0,
            "distance_ratio_min": 0.15,
            "p_candidate_min": 0.0,
            "competitor_max_out_deg": 1,
        }

        new_edges, _ = apply_dc_cmr(coords, edges, scale, reassign_cfg, cand_npz=None)
        lineage = Lineage(coords, new_edges)
        lineage.validate()  # Invariant checks: in_degree <= 1, out_degree <= 2, consecutive frames

        # Verify edge count delta is exactly 0
        self.assertEqual(len(new_edges), len(edges))

    def test_build_candidate_graph_033_includes_reassignment(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 20.0, 20.0],
            [0.0, 10.0, 15.0, 20.0],
            [1.0, 10.0, 25.0, 20.0],
            [1.0, 10.0, 15.0, 20.0],
        ], dtype=np.float64)
        edges = np.array([[0, 2], [1, 3]], dtype=np.int64)
        anchor = Lineage(coords, edges)

        cfg = config()["variants"]["variant_2"]
        cand, audit = build_candidate_graph(anchor, scale, None, {}, cfg, group="44b6")
        self.assertIn("smoothing", audit)
        self.assertIn("fork_pruning", audit)
        self.assertIn("mitosis_reassignment", audit)
        cand.validate()

    def test_load_cand_npz_paths(self):
        import tempfile
        from pathlib import Path
        from core.temporal_association_033 import _load_cand_npz, _load_cand_npz_confirm

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Create simulated structure for A1555 / A1688 runs:
            # base / runs / b0970-prepare-004 / cache-development / raw / test_candidates.npz
            prep4_dev = base / "b0970-prepare-004" / "cache-development" / "raw"
            prep4_dev.mkdir(parents=True)
            dummy_npz = prep4_dev / "sample_001_candidates.npz"
            np.savez(dummy_npz, candidate_edges=np.zeros((0, 2), dtype=int), candidate_probs=np.zeros((0,), dtype=float))

            source_path = base / "b0970-selector-corpus-011"
            source_path.mkdir(parents=True)

            res = _load_cand_npz("sample_001", source_path)
            self.assertIsNotNone(res)
            self.assertIn("candidate_edges", res)

            # Confirmation loader
            prep5_conf = base / "b0970-prepare-005" / "cache-confirmation" / "raw"
            prep5_conf.mkdir(parents=True)
            dummy_conf_npz = prep5_conf / "sample_conf_candidates.npz"
            np.savez(dummy_conf_npz, candidate_edges=np.zeros((0, 2), dtype=int), candidate_probs=np.zeros((0,), dtype=float))

            source_conf_path = base / "b0970-prepare-005"
            res_conf = _load_cand_npz_confirm("sample_conf", source_conf_path)
            self.assertIsNotNone(res_conf)
            self.assertIn("candidate_edges", res_conf)


if __name__ == "__main__":
    unittest.main()
