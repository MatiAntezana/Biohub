import json
from pathlib import Path
import unittest
import numpy as np

from core.temporal_association_030 import (
    config,
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
    build_candidate_graph,
    verify_sources,
)
from core.nextgen.contracts import Lineage


class TemporalAssociation030Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_030")
        self.assertEqual(cfg["scope"], "official_confirmation_evaluation_holdout")
        self.assertIn("variant", cfg)
        v = cfg["variant"]
        self.assertEqual(v["name"], "KP-CR-HGP-V1")
        self.assertTrue(v["smoothing"]["enabled"])
        self.assertTrue(v["smoothing"]["symmetric_only"])
        self.assertTrue(v["fork_pruning"]["enabled"])
        self.assertTrue(v["fork_pruning"]["heading_alignment"])
        self.assertTrue(v["fork_pruning"]["photometric_branch_selection"])

        gates = cfg["gates"]
        self.assertEqual(gates["confirmation_score_delta_min"], 0.002)

    def test_axial_z_smoothing_symmetric(self):
        coords = np.array([
            [0.0, 10.0, 20.0, 30.0],
            [1.0, 12.0, 23.0, 34.0],
            [2.0, 10.0, 26.0, 38.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        cfg_sm = {
            "enabled": True,
            "reversal_only": True,
            "symmetric_only": True,
            "jitter_threshold_um": 0.8,
            "max_correction_um": 0.6,
            "weight_z": 0.80,
            "weight_xy": 0.0
        }
        smoothed, audit = apply_harmonized_axial_regularizer(coords, edges, scale, cfg_sm)
        self.assertEqual(audit["spikes_suppressed"], 1)
        self.assertLess(smoothed[1, 1], coords[1, 1])

    def test_fork_pruning_safe_zero(self):
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 12.0, 10.0],
            [2.0, 10.0, 15.0, 10.0],
            [2.0, 10.0, 15.0, 11.0],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

        pr_cfg = {
            "enabled": True,
            "mode": "safe_zero_photometric",
            "cos_thresh": 0.00,
            "min_ratio": 0.18,
            "min_max_dist_um": 3.0,
            "heading_alignment": False,
            "photometric_branch_selection": False
        }
        retained, audit = prune_spurious_cytokinesis_forks(coords, edges, scale, None, pr_cfg)
        self.assertEqual(audit["forks_inspected"], 1)
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(len(retained), 2)

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
        vcfg = cfg["variant"]
        cand, audit = build_candidate_graph(anchor, scale, None, {}, vcfg, group="44b6")
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), len(anchor.coords))

    def test_verify_sources(self):
        with self.assertRaises(ValueError):
            verify_sources("/some/path/wrong-source-name", {})

        valid_dir = Path("artifacts/A1688/b0970-prepare-005")
        if valid_dir.is_dir():
            plan = verify_sources(valid_dir, {})
            self.assertIn("confirmation", plan["panels"])

    def test_remote_gt_location(self):
        import platform
        if platform.system() == "Linux":
            p = Path("/home/alumno1/Documents/Biohub/data/train")
            if p.is_dir():
                target = p / "44b6_2a2eff9f.geff"
                self.assertTrue(target.exists(), f"Expected {target} to exist")


if __name__ == "__main__":
    unittest.main()
