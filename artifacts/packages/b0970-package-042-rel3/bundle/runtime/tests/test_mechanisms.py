"""Synthetic mechanisms and solver constraints; these are not score evidence."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from core.nextgen.candidates import build_candidates, mutual_mapping
from core.nextgen.contracts import Lineage, write_submission
from core.nextgen.hoct_bridge import context_batches, blend_probabilities
from core.nextgen.solver import solve_lineage

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/nextgen.json").read_text())


class CandidateTests(unittest.TestCase):
    def test_neural_identity_mapping_reorders_without_guessing(self):
        coords = np.array([[0, 0, 0, 0], [0, 0, 0, 5], [1, 0, 0, 0]], float)
        self.assertEqual(mutual_mapping(coords, coords[[1, 2, 0]], np.ones(3), .1), {0: 1, 1: 2, 2: 0})
        duplicate = np.vstack([coords, coords[0] + [0, 0, 0, .01]])
        self.assertNotIn(0, mutual_mapping(coords, duplicate, np.ones(3), .1))

    def test_weak_prior_and_geometric_alternative_are_preserved(self):
        coords = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [1, 0, 0, 0]], float)
        graph = Lineage(coords, np.array([[0, 2]], np.int64))
        pairs, probabilities, _ = build_candidates(graph, coords, np.array([[0, 2, .01, 0]]), np.ones(3), CONFIG["candidates"])
        found = {tuple(edge): p for edge, p in zip(pairs, probabilities)}
        self.assertEqual(found[(0, 2)], .01)
        self.assertTrue(np.isnan(found[(1, 2)]))

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

    def test_context_refuses_silent_truncation(self):
        coords = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], float)
        with self.assertRaises(MemoryError):
            list(context_batches(coords, np.array([[0, 2], [1, 2]]), np.ones(3),
                 {**CONFIG["hoct"], "max_nodes_per_batch": 1}))

    def test_unknown_neural_prior_does_not_become_a_fake_score(self):
        result = blend_probabilities(np.array([.8, .8]), np.array([np.nan, .2]), .5)
        np.testing.assert_allclose(result, [.8, .5])

    def test_scaled_row_budget_enforcement(self):
        from core.submission_contract import validate_csv
        graph = Lineage(np.array([[0, 1, 1, 1], [1, 1, 1, 2]], float), np.array([[0, 1]], np.int64))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.csv"
            write_submission(path, {f"sample_{i}": graph for i in range(5)}, enforce_budget=False)
            with self.assertRaisesRegex(ValueError, "scaled row budget"):
                validate_csv(path, enforce_row_budget=True, expected_datasets=[f"sample_{i}" for i in range(5)])

    def test_hoct_uses_parental_softmax_with_orphan(self):
        import torch
        from core.nextgen.hoct_bridge import score_edges
        api = types.ModuleType("hoct._api")
        api.__file__ = "fake_audited_api.py"
        api._MEAN, api._STD = np.zeros(19), np.ones(19)
        package = types.ModuleType("hoct")
        package._api = api
        class Model:
            def eval(self):
                return self
            def __call__(self, features, positions, edge_positions, edges, nodes_mask, edges_mask):
                self.assertion = features.shape[-1] == 19
                return (torch.full((*edges.shape[:2], 1), float(np.log(2))), None, None,
                        torch.zeros((*features.shape[:2], 1)))
        coords = np.array([[0, 1, 1, 1], [0, 1, 2, 1], [1, 1, 1, 2]], float)
        spec = {"api_sha256": "audited", "checkpoint": "fake.pt", "sha256": "audited",
                "admissibility_reviewed": True, "source_url": "test-fixture"}
        with patch.dict("sys.modules", {"hoct": package, "hoct._api": api}), \
                patch("core.nextgen.hoct_bridge.verify_file"), patch("torch.jit.load", return_value=Model()):
            scores = score_edges(coords, np.array([[0, 2], [1, 2]]), np.ones((3, 19)),
                np.ones(3), spec, CONFIG["hoct"], device="cpu")
        np.testing.assert_allclose(scores, [.4, .4], atol=1e-6)


class SolverTests(unittest.TestCase):
    def test_second_child_requires_the_exact_approved_pair(self):
        coords = np.array([[0, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 1], [1, 2, 1, 1]], float)
        pairs = np.array([[0, 1], [0, 2], [0, 3]], np.int64)
        config = {**CONFIG["solver"], "division_penalty": 0, "disappearance_penalty": 0}
        graph, _ = solve_lineage(coords, pairs, [.99, .99, .999], [(0, 1, 2, .5)], 4, config)
        self.assertEqual(set(map(tuple, graph.edges)), {(0, 1), (0, 2)})
        np.testing.assert_array_equal(graph.coords, coords)

    def test_neural_scores_change_selected_parent(self):
        coords = np.array([[0, 1, 1, 1], [0, 1, 2, 1], [1, 1, 1, 2]], float)
        pairs = np.array([[0, 2], [1, 2]], np.int64)
        for probabilities, expected in (([.99, .01], (0, 2)), ([.01, .99], (1, 2))):
            graph, _ = solve_lineage(coords, pairs, probabilities, [], 3, CONFIG["solver"])
            self.assertEqual(set(map(tuple, graph.edges)), {expected})

    def test_no_approved_mitosis_means_maximum_one_child(self):
        coords = np.array([[0, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 1]], float)
        graph, _ = solve_lineage(coords, np.array([[0, 1], [0, 2]]), [.99, .98], [], 3, CONFIG["solver"])
        self.assertEqual(len(graph.edges), 1)

    def test_rescue_without_both_links_is_not_exported(self):
        coords = np.array([[0, 1, 1, 1], [1, 1, 1, 2]], float)
        graph, _ = solve_lineage(coords, np.array([[0, 1]]), [.999], [], 1, CONFIG["solver"])
        self.assertEqual(len(graph.coords), 1)
        self.assertEqual(len(graph.edges), 0)

    def test_small_diagnostic_csv_is_not_a_submission(self):
        graph = Lineage(np.array([[0, 1, 1, 1], [1, 1, 1, 2]], float), np.array([[0, 1]], np.int64))
        with tempfile.TemporaryDirectory() as folder:
            report = write_submission(Path(folder) / "diagnostic.csv", {"sample": graph}, enforce_budget=False)
            self.assertEqual(report["csv_rows"], 3)
            with self.assertRaisesRegex(ValueError, "row budget"):
                write_submission(Path(folder) / "candidate.csv", {"sample": graph}, enforce_budget=True)


class MitosisTests(unittest.TestCase):
    def test_missing_annotations_are_not_negative_labels(self):
        from core.nextgen.mitosis import confirmed_label
        self.assertIsNone(confirmed_label(1, 2, 3, {2: 1}))
        self.assertEqual(confirmed_label(1, 2, 3, {2: 1, 3: 1}), 1)
        self.assertEqual(confirmed_label(1, 2, 3, {2: 1, 3: 4}), 0)

    def test_daughter_order_does_not_change_head_inputs(self):
        from core.nextgen.mitosis import example_tensors
        class Source:
            scale = np.array([1.625, .40625, .40625])
            shape = (6, 50, 50, 50)
            def patch(self, t, centre, half_um, shape):
                return np.full(shape, t, np.float32)
        triple = np.array([[2, 10, 10, 10], [3, 10, 10, 8], [3, 11, 10, 12]], float)
        left, right = example_tensors(triple, Source()), example_tensors(triple[[0, 2, 1]], Source())
        np.testing.assert_allclose(left[0], right[0])
        np.testing.assert_allclose(left[1], right[1])


class ImageEvidenceTests(unittest.TestCase):
    def source(self, dark_middle=False):
        from core.nextgen.images import ImageSource
        class Source(ImageSource):
            def __init__(self):
                self.scale = np.ones(3)
                self.shape = (5, 25, 25, 25)
            def frame(self, t):
                if dark_middle and t == 2:
                    return np.zeros(self.shape[1:], np.float32)
                grid = np.indices(self.shape[1:]).astype(float)
                centre = np.array([12, 12, [6, 7, 9, 10, 11][t]])[:, None, None, None]
                return np.exp(-((grid - centre) ** 2).sum(axis=0) / 2).astype(np.float32)
        return Source()

    def test_features_require_measured_foreground(self):
        from core.nextgen.images import measured_features
        coords = np.array([[2, 12, 12, 9]], float)
        features = measured_features(coords, self.source())
        self.assertEqual(features.shape, (1, 19))
        self.assertGreater(features[0, 4], 0)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            measured_features(coords, self.source(dark_middle=True))

    def test_rescue_follows_image_peak_not_midpoint(self):
        from core.nextgen.rescue import propose_rescues
        graph = Lineage(np.array([[0, 12, 12, 6], [1, 12, 12, 7], [3, 12, 12, 10], [4, 12, 12, 11]], float),
                        np.array([[0, 1], [2, 3]], np.int64))
        config = {**CONFIG["rescue"], "max_fraction": 1.0}
        proposals = propose_rescues(graph, self.source(), config)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["coord"], [2, 12, 12, 9])
        self.assertNotEqual(proposals[0]["coord"][-1], 8.5)
        self.assertEqual(propose_rescues(graph, self.source(dark_middle=True), config), [])


if __name__ == "__main__":
    unittest.main()
