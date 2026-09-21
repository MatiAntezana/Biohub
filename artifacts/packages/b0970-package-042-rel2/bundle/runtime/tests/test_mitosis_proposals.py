"""The new proposal pool must cover blocked forks without touching their graph."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.artifacts import digest, dump
from core.mitosis_probe import GEOMETRY_NAMES
from core.mitosis_proposals import (ADDITIONS, DRIVERS, SCOPE, config, verify_sources,
    geometry_features, enumerate_pairs, score_pairs, topology_maps, edit_footprint,
    label_pairs, annotation_summary, fork_coverage, tree_fingerprint, proposal_job,
    predict_dataset)
from core.nextgen.contracts import Lineage
from core.nextgen.mitosis import example_tensors


def fixture():
    anchor = Lineage(np.array([[0, 10, 10, 10], [0, 10, 10, 14],
        [1, 10, 10, 9], [1, 10, 10, 11], [1, 10, 10, 15], [1, 10, 10, 17]], float),
        np.array([[0, 2], [1, 3]], np.int64))
    mapping = {10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
    children, parents = {10: {12, 13}, 11: {14}}, {12: 10, 13: 10, 14: 11}
    gt = {10: anchor.coords[[0, 2, 3]].copy()}
    return anchor, mapping, children, parents, gt


def toy_model():
    return dict(variant="geometry", feature_names=list(GEOMETRY_NAMES),
        standardizer=dict(mean=[0.]*11, scale=[1.]*11, active=[True]*11),
        weights=[.1]*11, intercept=0.)


class MitosisProposalsTests(unittest.TestCase):
    def test_geometry_matches_frozen_feature_formula_and_daughter_permutation(self):
        rng = np.random.default_rng(42)
        triples = rng.uniform(10, 30, (12, 3, 4))
        triples[:, :, 0] = [0, 1, 1]
        triples[4:8, :, 0] = [4, 5, 5]
        triples[8:, :, 0] = [8, 9, 9]
        source = SimpleNamespace(scale=np.array([1.625, .40625, .40625]), shape=(10, 64, 64, 64),
            patch=lambda *a, **kw: np.zeros((9, 25, 25), np.float32))
        expected = np.stack([example_tensors(t, source)[1] for t in triples])
        actual = geometry_features(triples, source.scale, 10)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(actual, geometry_features(triples[:, [0, 2, 1]], source.scale, 10))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual[0, 7:], [0, 1, 1, 1])
        np.testing.assert_array_equal(actual[-1, 7:], [1, 1, 1, 0])

    def test_features_reject_invalid_time_and_accept_empty_batch(self):
        triple = fixture()[0].coords[[0, 2, 3]][None]
        for t in (2, -.5, 1.5):
            bad = triple.copy(); bad[0, 2, 0] = t
            with self.assertRaises(ValueError):
                geometry_features(bad, [1, 1, 1], 10)
        with self.assertRaises(ValueError):
            geometry_features(triple, [0, 1, 1], 10)
        self.assertEqual(geometry_features(np.empty((0, 3, 4)), [1, 1, 1], 10).shape, (0, 11))

    def test_enumeration_includes_occupied_daughters_without_priors_and_keeps_graph(self):
        anchor, *_ = fixture()
        before_c, before_e = anchor.coords.copy(), anchor.edges.copy()
        triples = enumerate_pairs(anchor, [1, 1, 1], config())
        self.assertEqual(len(triples), 12)
        self.assertIn((0, 2, 3), set(map(tuple, triples)))
        self.assertTrue((triples[:, 1] < triples[:, 2]).all())
        np.testing.assert_array_equal(anchor.coords, before_c)
        np.testing.assert_array_equal(anchor.edges, before_e)

    def test_top_six_uses_distance_then_id_including_ties(self):
        coords = [[0, 10, 10, 10]] + [[1, 10, 10, 10+i] for i in range(1, 8)]
        # Two equidistant daughters at the cutoff: deterministic lower ID wins.
        coords[-1][-1] = 4
        anchor = Lineage(np.array(coords, float), np.empty((0, 2), np.int64))
        triples = enumerate_pairs(anchor, [1, 1, 1], config())
        self.assertEqual(len(triples), 15)
        self.assertIn(6, triples[:, 1:].ravel())
        self.assertNotIn(7, triples[:, 1:].ravel())

    def test_radius_is_physical_and_no_gap_time_pairs_are_generated(self):
        c = np.array([[0, 10, 10, 10], [1, 19, 10, 10], [1, 10, 19, 10],
            [1, 10, 10, 19], [2, 10, 10, 10]], float)
        anchor = Lineage(c, np.empty((0, 2), np.int64))
        triples = enumerate_pairs(anchor, [1.625, .40625, .40625], config())
        np.testing.assert_array_equal(triples, [[0, 2, 3]])

    def test_budget_failure_is_not_silent_truncation_and_recipe_is_frozen(self):
        cfg = {**config(), "max_pairs_per_dataset": 1}
        with patch("core.mitosis_proposals.config", return_value=cfg), self.assertRaisesRegex(ValueError, "budget"):
            enumerate_pairs(fixture()[0], [1, 1, 1], cfg)
        with self.assertRaisesRegex(ValueError, "fixed"):
            enumerate_pairs(fixture()[0], [1, 1, 1], {**config(), "max_children": 10})

    def test_scoring_is_batched_and_has_no_image_or_label_input(self):
        anchor, *_ = fixture()
        triples = enumerate_pairs(anchor, [1, 1, 1], config())
        a = score_pairs(anchor, triples, [1, 1, 1], 10, toy_model(), 2)
        b = score_pairs(anchor, triples, [1, 1, 1], 10, toy_model(), 100)
        np.testing.assert_allclose(a, b)
        self.assertEqual(score_pairs(anchor, np.empty((0, 3), int), [1, 1, 1], 10, toy_model(), 2).shape, (0,))
        with self.assertRaises(ValueError):
            score_pairs(anchor, triples, [1, 1, 1], 10, {**toy_model(), "variant": "frozen_image"}, 2)

    def test_edit_footprint_describes_reassignment_without_applying_it(self):
        anchor, *_ = fixture()
        children, parents = topology_maps(anchor)
        self.assertEqual(edit_footprint((0, 2, 3), children, parents),
            dict(kind="requires_reassignment", remove_edges=[(1, 3)], add_edges=[(0, 3)]))
        self.assertEqual(edit_footprint((0, 4, 5), children, parents)["remove_edges"], [(0, 2)])
        self.assertEqual(edit_footprint((0, 2, 5), children, parents)["kind"], "additive_second_child")
        self.assertEqual(edit_footprint((9, 4, 5), children, parents)["kind"], "new_two_child_fork")
        self.assertEqual(edit_footprint((0, 2, 3), {0: {2, 3}}, {2: 0, 3: 0})["kind"], "already_present")
        np.testing.assert_array_equal(anchor.edges, [[0, 2], [1, 3]])

    def test_unknown_annotation_is_never_a_negative(self):
        _, mapping, _, parents, _ = fixture()
        triples = np.array([[0, 2, 3], [0, 2, 4], [0, 2, 5], [1, 2, 3]])
        np.testing.assert_array_equal(label_pairs(triples, mapping, parents), [1, 0, -1, 0])
        with self.assertRaises(ValueError):
            label_pairs(triples, {**mapping, 99: 0}, parents)
        report = annotation_summary([1, 0, -1], [2, -2, 9], .8)
        self.assertEqual(report["unknown_at_frozen_threshold"], 1)
        self.assertEqual(report["confirmed_only_metrics"]["fp"], 0)
        self.assertIsNone(annotation_summary([-1], [1], .8)["confirmed_only_metrics"]["auroc"])
        with self.assertRaises(ValueError):
            annotation_summary([-1], [np.nan], .8)

    def test_coverage_reports_legacy_gap_topology_and_tie_rank(self):
        anchor, mapping, children, _, gt = fixture()
        triples = np.array([[0, 2, 3], [0, 2, 4], [0, 3, 4]])
        rows = fork_coverage(anchor, triples, np.array([1., 2., 1.]), [], mapping, children, gt, np.ones(3), config())
        self.assertTrue(rows[0]["covered"])
        self.assertFalse(rows[0]["legacy_scored"])
        self.assertEqual((rows[0]["rank_best"], rows[0]["rank_worst"]), (2, 3))
        self.assertEqual(rows[0]["edit_footprint"]["kind"], "requires_reassignment")

    def test_coverage_separates_unmatched_radius_topk_and_nonconsecutive(self):
        anchor, mapping, children, _, gt = fixture()
        def coverage(mapping, scale, gt=gt):
            return fork_coverage(anchor, np.empty((0, 3), int), np.empty(0), [], mapping, children, gt, np.array(scale), config())[0]
        self.assertEqual(coverage({10: 0}, [1, 1, 1])["reason"], "unmatched_endpoint")
        self.assertEqual(coverage(mapping, [1, 1, 1])["reason"], "outside_nearest_children")
        self.assertEqual(coverage(mapping, [1, 1, 13])["reason"], "outside_radius")
        changed = gt[10].copy(); changed[2, 0] = 2
        self.assertEqual(coverage(mapping, [1, 1, 1], {10: changed})["reason"], "nonconsecutive_gt_fork")

    def test_sources_enforce_previous_science_and_asset_pins(self):
        cfg = {**config(), "parent_science_sha256": hashlib.sha256(b"{}").hexdigest()}
        manifest = dict.fromkeys(DRIVERS | ADDITIONS, "new")
        with patch("core.mitosis_proposals.config", return_value=cfg), patch("core.mitosis_proposals.verify_probe_source"), patch("core.mitosis_proposals.verify_file", side_effect=ValueError("changed asset")):
            with self.assertRaisesRegex(ValueError, "changed asset"):
                verify_sources(Path("b0970-prepare-005"), manifest)
            with self.assertRaisesRegex(ValueError, "exact 008"):
                verify_sources(Path("b0970-prepare-005"), {**manifest, "core/extra.py": "new"})
            with self.assertRaisesRegex(ValueError, "exact 008"):
                verify_sources(Path("b0970-probe-008"), manifest)

    def test_annotation_fingerprint_detects_changed_or_empty_stores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                tree_fingerprint(root)
            (root / "chunk").write_bytes(b"abc")
            before = tree_fingerprint(root)
            (root / "chunk").write_bytes(b"def")
            self.assertNotEqual(before, tree_fingerprint(root))

    def test_prediction_uses_opposite_probe_metadata_only_and_no_gt(self):
        anchor, *_ = fixture()
        name = "44b6_dev"
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "b0970-prepare-005"
            dump(source / "cache-development/raw" / (name + ".json"), dict(candidates_sha256="f"*64))
            with patch("core.mitosis_proposals.load_reference", return_value={name: anchor}), patch("core.nextgen.images.ImageSource", return_value=SimpleNamespace(shape=(10, 64, 64, 64))) as image, patch("core.mitosis_proposals.load_probe", return_value=toy_model()) as loader, patch("core.mitosis_proposals.load_candidates", return_value=(anchor.coords.copy(), np.empty((0, 4)))), patch("core.mitosis_proposals.match_annotation", side_effect=AssertionError("GT forbidden")):
                result = predict_dataset(source, name, dict(data_root="images"), config())
            self.assertEqual(loader.call_args.args[1:], ("44b6", "6bba", "geometry"))
            self.assertTrue(loader.call_args.args[0]["path"].endswith("b0970-probe-008/train-6bba/geometry.json"))
            self.assertEqual(image.call_args.args[1:], ("0", [1.625, .40625, .40625]))
            self.assertEqual(result[-1]["geometric_pairs"], 12)
            self.assertEqual(result[-1]["probe_trained_on_group"], "6bba")

    def test_job_freezes_all_predictions_before_gt_and_never_publishes(self):
        anchor, mapping, children, parents, gt = fixture()
        names = ["44b6_dev", "6bba_dev"]
        events = []
        def predict(source, name, plan, cfg):
            events.append(name)
            return anchor, np.array([[0, 2, 3]]), np.array([2.]), np.empty((0, 3), np.int64), {
                "dataset": name, "evaluated_data_group": name[:4], "scale_um": [1, 1, 1]}
        def match(*args):
            self.assertEqual(events[:2], sorted(names))
            self.assertTrue((root / "prediction-manifest.json").exists())
            events.append("annotation")
            return mapping, children, parents, gt
        with tempfile.TemporaryDirectory() as folder:
            root, source = Path(folder) / "run", Path(folder) / "source"
            root.mkdir(); source.mkdir()
            dump(source / "plan.json", dict(panels=dict(development=names), data_root="unused"))
            with patch("core.mitosis_proposals.PROTOCOL", dict(development=names)), patch("core.mitosis_proposals.verify_sources"), patch("core.mitosis_proposals.predict_dataset", side_effect=predict), patch("core.mitosis_proposals.load_reference", side_effect=lambda path, ns: {n: anchor for n in ns}), patch("core.mitosis_proposals.tree_fingerprint", return_value="f"*64), patch("core.mitosis_proposals.match_annotation", side_effect=match):
                result = proposal_job(SimpleNamespace(), root, source)
            self.assertEqual(result["disposition"], SCOPE)
            for k in ("graph_modified", "tracking_evaluated", "confirmation_evaluated", "threshold_selected"):
                self.assertFalse(result[k])
            self.assertFalse((root / "submission.csv").exists())
            report = json.loads((root / "proposals.json").read_text())
            self.assertEqual(report["by_group"]["44b6"]["covered_nonpresent_gt_forks"], 1)
            with np.load(root / "44b6_dev.proposals.npz") as arrays:
                self.assertNotIn("labels", arrays.files)
            np.testing.assert_array_equal(anchor.edges, [[0, 2], [1, 3]])

    def test_proposals_require_current_tests_and_cannot_feed_tracking(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "runs"
            dump(root / "source/receipt.json", dict(task="prepare", status="completed", code_hash="old"))
            checked = dict(task="tests", status="completed", code_hash="new", result=dict(tests_passed=True))
            dump(root / "tests/receipt.json", checked)
            args = SimpleNamespace(task="propose-mitosis", artifact_root=folder, code_hash="new", source_run="source", tests_run="tests")
            with patch("core.mitosis_proposals.verify_sources"):
                self.assertEqual(source_run(args), root / "source")
                dump(root / "tests/receipt.json", {**checked, "code_hash": "old"})
                with self.assertRaises(ValueError):
                    source_run(args)
            dump(root / "source/receipt.json", dict(task="propose-mitosis", status="completed", code_hash="new"))
            for task in ("experiment", "prepare", "package"):
                args.task = task
                with self.assertRaisesRegex(ValueError, "Expected source task"):
                    source_run(args)


class MitosisProposalsRuntimeTests(unittest.TestCase):
    def test_official_matching_labels_the_occupied_true_fork_and_coverage(self):
        from core.evaluation import make_graph
        anchor, *_ = fixture()
        gt = make_graph({0: [dict(id=10, pos=[10, 10, 10]), dict(id=11, pos=[10, 10, 14])],
            1: [dict(id=12, pos=[10, 10, 9]), dict(id=13, pos=[10, 10, 11]), dict(id=14, pos=[10, 10, 15])]},
            [(10, 12), (10, 13), (11, 14)])
        from core.mitosis_diagnostic import match_annotation
        with patch("ops.workflow.load_graph", return_value=gt):
            mapping, children, parents, coords = match_annotation(anchor, "unused", [1, 1, 1])
        pairs = enumerate_pairs(anchor, [1, 1, 1], config())
        labels = label_pairs(pairs, mapping, parents)
        self.assertEqual(int((labels == 1).sum()), 1)
        self.assertTrue((labels == -1).any())
        rows = fork_coverage(anchor, pairs, np.zeros(len(pairs)), [], mapping, children, coords, np.ones(3), config())
        self.assertTrue(rows[0]["covered"])
        self.assertEqual(rows[0]["edit_footprint"]["kind"], "requires_reassignment")


if __name__ == "__main__":
    unittest.main()
