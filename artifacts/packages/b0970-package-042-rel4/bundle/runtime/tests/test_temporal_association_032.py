"""Unit tests for Campaign 032 Distance-Dominant Keeper Selection & Spatial Proximity Invariant (DDKS-SPI)."""
from pathlib import Path
import unittest
import numpy as np

from core.nextgen.association import Lineage
from core.temporal_association_032 import (
    config,
    verify_sources,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
)


class TemporalAssociation032Tests(unittest.TestCase):

    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_032")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertIn("variant_3", cfg["variants"])
        self.assertIn("gates", cfg)

        v1 = cfg["variants"]["variant_1"]["fork_pruning"]
        self.assertTrue(v1["spi_enabled"])
        self.assertEqual(v1["distance_cost_mode"], "linear")
        self.assertEqual(v1["heading_guard_cos_max"], 1.0)

        v2 = cfg["variants"]["variant_2"]["fork_pruning"]
        self.assertTrue(v2["spi_enabled"])
        self.assertEqual(v2["distance_cost_mode"], "linear")
        self.assertEqual(v2["heading_guard_cos_max"], 0.0)

        v3 = cfg["variants"]["variant_3"]["fork_pruning"]
        self.assertTrue(v3["spi_enabled"])
        self.assertEqual(v3["distance_cost_mode"], "quadratic")
        self.assertEqual(v3["heading_guard_cos_max"], 0.2)

        self.assertIn("variant_4", cfg["variants"])
        v4 = cfg["variants"]["variant_4"]["fork_pruning"]
        self.assertTrue(v4["spi_enabled"])
        self.assertEqual(v4["distance_cost_mode"], "quadratic")
        self.assertEqual(v4["heading_guard_cos_max"], 1.0)

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

    def test_spatial_proximity_invariant_overrides_velocity_alignment(self):
        # Scenario reproducing 44b6_2a2eff9f edge TP loss:
        # Mother had incoming velocity in +y direction.
        # d1 is a true stationary cell at 0.8 um (disp [0, 2, 0] pix = 0.81 um).
        # d2 is a jumping crossover at 5.5 um in +y direction (align=+1.0).
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [-1.0, 10.0, 5.0, 10.0],   # p: parent at t=-1
            [0.0, 10.0, 10.0, 10.0],   # m: mother at t=0 (incoming velocity +5 pix in y = 2.03 um)
            [1.0, 10.0, 12.0, 10.0],   # d1: stationary cell (+2 pix = 0.81 um)
            [1.0, 10.0, 24.0, 10.0],   # d2: jumping crossover (+14 pix = 5.68 um in +y direction)
        ], dtype=np.float64)
        edges = np.array([
            [0, 1],
            [1, 2],
            [1, 3],
        ], dtype=np.int64)

        # Under SPI (Variant 1):
        pr_cfg_spi = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "spi_max_close_um": 1.5,
            "spi_min_far_um": 2.5,
            "spi_min_ratio": 2.0,
            "distance_cost_mode": "linear",
            "heading_alignment": True,
            "photometric_branch_selection": False,
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg_spi)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(audit["spi_triggered"], 1)
        retained_set = set(map(tuple, retained))
        # d1 (close daughter) MUST be retained!
        self.assertIn((1, 2), retained_set)
        # d2 (jumping daughter) MUST be pruned!
        self.assertNotIn((1, 3), retained_set)

    def test_dual_track_biological_division_protected(self):
        # Scenario reproducing 6bba_48816121 mother 834:
        # Mother m divides into d1 (3.6 um) and d2 (0.4 um).
        # Even though spatial asymmetry is extreme (0.4 vs 3.6 um),
        # both d1 and d2 have persistent forward tracks (len >= 2, head > 0.3).
        # The biological dual-track protection guard MUST prevent either branch from being pruned!
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],   # m (0)
            [1.0, 10.0, 19.0, 10.0],   # d1 (1): +9 pix y = 3.65 um
            [1.0, 10.0, 11.0, 10.0],   # d2 (2): +1 pix y = 0.41 um
            [2.0, 10.0, 28.0, 10.0],   # g1 (3): continues forward
            [3.0, 10.0, 37.0, 10.0],   # gg1 (4)
            [2.0, 10.0, 12.0, 10.0],   # g2 (5): continues forward
            [3.0, 10.0, 13.0, 10.0],   # gg2 (6)
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [0, 2],
            [1, 3], [3, 4],
            [2, 5], [5, 6],
        ], dtype=np.int64)

        pr_cfg_spi = {
            "enabled": True,
            "mode": "spatial_proximity_invariant",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": True,
            "spi_max_close_um": 1.5,
            "spi_min_far_um": 2.5,
            "spi_min_ratio": 2.0,
            "distance_cost_mode": "linear",
            "heading_alignment": True,
            "photometric_branch_selection": False,
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg_spi)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 0)
        self.assertEqual(audit["spi_triggered"], 0)
        retained_set = set(map(tuple, retained))
        # BOTH branches must be protected!
        self.assertIn((0, 1), retained_set)
        self.assertIn((0, 2), retained_set)

    def test_quadratic_distance_cost_penalizes_large_jumps(self):
        # Even without SPI, Quadratic Distance Cost (-1.0 * (d/2.0)^2) penalizes
        # 5.68 um by -8.08 points, easily beating alignment of +1.0 point!
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [-1.0, 10.0, 5.0, 10.0],   # p
            [0.0, 10.0, 10.0, 10.0],   # m
            [1.0, 10.0, 12.0, 10.0],   # d1: 0.81 um
            [1.0, 10.0, 24.0, 10.0],   # d2: 5.68 um
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)

        pr_cfg_qdc = {
            "enabled": True,
            "mode": "quadratic_distance_cost",
            "cos_thresh": 0.0,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "spi_enabled": False,
            "distance_cost_mode": "quadratic",
            "distance_scale_d0_um": 2.0,
            "distance_cost_alpha": 1.0,
            "heading_alignment": True,
            "photometric_branch_selection": False,
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg_qdc)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        retained_set = set(map(tuple, retained))
        self.assertIn((1, 2), retained_set)
        self.assertNotIn((1, 3), retained_set)

    def test_build_candidate_graph_integrity(self):
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 10.5, 10.0],
            [1.0, 10.0, 15.0, 10.0],
            [2.0, 10.0, 11.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
        anchor = Lineage(coords, edges)

        v_cfg = {
            "smoothing": {"enabled": False},
            "fork_pruning": {
                "enabled": True,
                "mode": "spatial_proximity_invariant",
                "cos_thresh": 0.0,
                "min_ratio": 0.18,
                "min_max_dist_um": 3.0,
                "spi_enabled": True,
                "spi_max_close_um": 1.5,
                "spi_min_far_um": 2.5,
                "spi_min_ratio": 2.0,
                "distance_cost_mode": "linear",
                "heading_alignment": True,
                "photometric_branch_selection": False,
            }
        }
        cand, audit = build_candidate_graph(anchor, scale, None, {}, v_cfg, group="44b6")
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), len(anchor.coords))
        self.assertLess(len(cand.edges), len(anchor.edges))

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1555/b0970-selector-corpus-011")
        if valid_dir.is_dir():
            corpus = verify_sources(valid_dir, {})
            self.assertIn("development", corpus)

        conf_dir = Path("artifacts/A1688/b0970-prepare-005")
        if conf_dir.is_dir():
            plan = verify_sources(conf_dir, {})
            self.assertIn("confirmation", plan["panels"])
            self.assertEqual(len(plan["panels"]["confirmation"]), 8)

    def test_infer_temporal_panel_toy(self):
        import tempfile
        from ops.workflow import infer_temporal_panel
        from core.nextgen.contracts import write_submission
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()
            out_dir = tmp_path / "out"

            coords = np.array([
                [0.0, 10.0, 10.0, 10.0],
                [1.0, 10.0, 11.0, 10.0],
            ], dtype=np.float64)
            edges = np.array([[0, 1]], dtype=np.int64)
            toy_graph = Lineage(coords, edges)
            write_submission(cache_dir / "reference.csv", {"44b6_sample": toy_graph}, enforce_budget=False)

            cfg = config()
            result = infer_temporal_panel(
                cache_dir,
                ["44b6_sample"],
                cfg["variants"]["variant_2"],
                out_dir,
                scope="diagnostic",
            )
            self.assertIn("candidate.csv", [p.name for p in out_dir.glob("*")])
            self.assertIn("result.json", [p.name for p in out_dir.glob("*")])
            self.assertEqual(result["contract"]["csv_rows"], 3)
            self.assertEqual(result["contract"]["nodes"], 2)
            self.assertEqual(result["contract"]["edges"], 1)


if __name__ == "__main__":
    unittest.main()
