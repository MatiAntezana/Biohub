"""Unit tests for candidate generation, temporal context batching, and lineage contracts."""
import json
from pathlib import Path
import unittest
import numpy as np

from src.tracking.candidates import build_candidates, mutual_mapping
from src.tracking.contracts import Lineage
from src.tracking.hoct_bridge import blend_probabilities, context_batches

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "assets/nextgen.json").read_text())


class TrackingPipelineTests(unittest.TestCase):
    def test_neural_identity_mapping_reorders_without_guessing(self):
        coords = np.array([[0, 0, 0, 0], [0, 0, 0, 5], [1, 0, 0, 0]], float)
        self.assertEqual(mutual_mapping(coords, coords[[1, 2, 0]], np.ones(3), 0.1), {0: 1, 1: 2, 2: 0})
        duplicate = np.vstack([coords, coords[0] + [0, 0, 0, 0.01]])
        self.assertNotIn(0, mutual_mapping(coords, duplicate, np.ones(3), 0.1))

    def test_anisotropy_is_physical_not_voxel_radius(self):
        coords = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], float)
        graph = Lineage(coords, np.empty((0, 2), np.int64))
        config = {**CONFIG["candidates"], "radius_um": 1.5}
        pairs, _, _ = build_candidates(graph, coords, np.empty((0, 4)), np.array([2, 1, 1]), config)
        np.testing.assert_array_equal(pairs, [[1, 2]])

    def test_context_covers_all_edges_once_at_boundaries(self):
        coords = np.asarray([[t, 1, 1, x] for t in range(5) for x in (1, 23, 25, 48)], float)
        pairs = np.asarray([(t * 4 + i, (t + 1) * 4 + j) for t in range(4) for i in range(4) for j in range(4)], np.int64)
        counts = np.zeros(len(pairs), int)
        for nodes, edges, owned in context_batches(coords, pairs, np.ones(3), CONFIG["hoct"]):
            self.assertLessEqual(len(np.unique(coords[nodes, 0])), 5)
            self.assertTrue(set(pairs[edges].ravel()).issubset(set(nodes)))
            counts[edges[owned]] += 1
        np.testing.assert_array_equal(counts, np.ones(len(pairs)))

    def test_unknown_neural_prior_does_not_become_a_fake_score(self):
        result = blend_probabilities(np.array([0.8, 0.8]), np.array([np.nan, 0.2]), 0.5)
        np.testing.assert_allclose(result, [0.8, 0.5])


if __name__ == "__main__":
    unittest.main()
