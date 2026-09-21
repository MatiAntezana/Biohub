"""Tests for 016 temporal association and anisotropic smoothing."""
import json
import unittest
import numpy as np

from core.artifacts import ROOT, code_manifest
from core.nextgen.contracts import Lineage
from core.temporal_association_016 import (
    config,
    parent_manifest,
    smooth_linear_tracks_anisotropic,
    resolve_edge_switches,
    build_candidate_graph,
    ADDITIONS,
    DRIVERS
)


class TemporalAssociation016Tests(unittest.TestCase):
    def test_config_schema(self):
        cfg = config()
        self.assertEqual(cfg["schema"], "temporal_association_016")
        self.assertIn("variant_1", cfg["variants"])
        self.assertIn("variant_2", cfg["variants"])
        self.assertEqual(cfg["node"], "A1555")
        self.assertGreater(cfg["gates"]["development_score_delta_min"], 0.0)

    def test_smooth_linear_tracks_anisotropic_preserves_endpoints_and_reduces_jitter(self):
        # 10 frames, linear motion in XY, jitter in Z
        # Frame t: t in 0..9
        t = np.arange(10, dtype=np.float64)
        z = 10.0 + t * 0.5
        # Add jitter spike at node 4 (frame 4)
        z[4] += 3.0  # ~4.8 um jitter spike in Z
        y = 20.0 + t * 1.0
        x = 30.0 + t * 1.0

        coords = np.column_stack([t, z, y, x])
        edges = np.column_stack([np.arange(9), np.arange(1, 10)]).astype(np.int64)
        scale = np.array([1.625, 0.40625, 0.40625])

        smoothed = smooth_linear_tracks_anisotropic(coords, edges, scale, weight_z=0.85, weight_xy=0.65, window=2)

        # Endpoints 0 and 9 must NOT be modified
        np.testing.assert_array_equal(smoothed[0], coords[0])
        np.testing.assert_array_equal(smoothed[9], coords[9])

        # Node 4 must have reduced Z jitter
        orig_jitter = abs(coords[4, 1] - (coords[3, 1] + coords[5, 1]) / 2.0)
        new_jitter = abs(smoothed[4, 1] - (smoothed[3, 1] + smoothed[5, 1]) / 2.0)
        self.assertLess(new_jitter, orig_jitter)

        # Ensure all coordinates remain unique
        self.assertEqual(len({tuple(row) for row in smoothed}), len(smoothed))

    def test_resolve_edge_switches_corrects_crossing(self):
        scale = np.array([1.625, 0.40625, 0.40625])
        # Two tracks moving from t=0, 1, 2
        # Track A: (0, 10, 10, 10) -> (1, 10, 11, 10) -> (2, 10, 12, 10)
        # Track B: (0, 10, 10, 14) -> (1, 10, 11, 14) -> (2, 10, 12, 14)
        # But edges at t=1 -> t=2 were swapped erroneously:
        # A(t=1) -> B(t=2) and B(t=1) -> A(t=2)
        coords = np.array([
            [0, 10, 10, 10],  # 0: A0
            [0, 10, 10, 14],  # 1: B0
            [1, 10, 11, 10],  # 2: A1
            [1, 10, 11, 14],  # 3: B1
            [2, 10, 12, 10],  # 4: A2
            [2, 10, 12, 14],  # 5: B2
        ], dtype=np.float64)

        # Erroneous swap at t=1: 2 -> 5 (cross), 3 -> 4 (cross)
        erroneous_edges = np.array([
            [0, 2], [1, 3],
            [2, 5], [3, 4]
        ], dtype=np.int64)

        # Descriptors: A has patch [1, 0], B has patch [0, 1]
        desc = np.zeros((6, 4), dtype=np.float32)
        desc[[0, 2, 4]] = [1.0, 0.0, 0.0, 0.0]  # Track A identity
        desc[[1, 3, 5]] = [0.0, 1.0, 0.0, 0.0]  # Track B identity
        stats = np.ones((6, 3), dtype=np.float32)

        cfg_switches = {
            "enabled": True,
            "search_radius_um": 6.0,
            "min_res_gain_um": 0.5,
            "min_ncc": 0.5
        }

        fixed_edges, audit = resolve_edge_switches(coords, erroneous_edges, scale, desc, stats, cfg_switches)

        self.assertEqual(audit["swaps_applied"], 1)
        fixed_set = set(map(tuple, fixed_edges))
        self.assertIn((2, 4), fixed_set)
        self.assertIn((3, 5), fixed_set)
        self.assertNotIn((2, 5), fixed_set)
        self.assertNotIn((3, 4), fixed_set)
        self.assertEqual(len(fixed_edges), len(erroneous_edges))

    def test_build_candidate_graph_validates(self):
        coords = np.array([
            [0, 10, 10, 10],
            [1, 10, 11, 10],
            [2, 10, 12, 10],
            [3, 10, 13, 10]
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        scale = np.array([1.625, 0.40625, 0.40625])
        desc = np.ones((4, 8), dtype=np.float32) / np.sqrt(8)
        stats = np.ones((4, 3), dtype=np.float32)

        anchor = Lineage(coords, edges)
        vcfg = {
            "smoothing": {"enabled": True, "weight_z": 0.85, "weight_xy": 0.65, "window": 2},
            "switches": {"enabled": False}
        }
        cand, audit = build_candidate_graph(anchor, scale, desc, stats, vcfg)
        self.assertIsInstance(cand, Lineage)
        self.assertEqual(len(cand.coords), 4)
        self.assertEqual(len(cand.edges), 3)


if __name__ == "__main__":
    unittest.main()
