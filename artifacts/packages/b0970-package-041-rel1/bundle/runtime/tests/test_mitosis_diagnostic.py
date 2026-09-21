"""Audit must observe the frozen model, not change or promote its hypotheses."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.mitosis_diagnostic import (capture_scores, classify_divisions, inventory,
    score_summary, training_batch_summary, verify_diagnostic_source)
from core.nextgen.contracts import Lineage
from core.nextgen import mitosis


def fixture():
    anchor = Lineage(np.array([[0, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 1]], float),
        np.array([[0, 1]], dtype=np.int64))
    config = dict(min_edge_probability=.05, radius_um=12., max_children=6, batch_size=2, threshold=.80)
    return anchor, np.array([[0, 1], [0, 2]]), np.array([.9, .2]), config


class MitosisDiagnosticTests(unittest.TestCase):
    def test_inventory_counts_pairs_and_free_daughters_without_threshold_sweep(self):
        anchor, pairs, priors, config = fixture()
        triples, addable = inventory(anchor, pairs, priors, [1, 1, 1], config)
        self.assertEqual(triples, [(0, 1, 2)])
        self.assertEqual(addable, {(0, 1, 2)})
        triples, _ = inventory(anchor, pairs, [np.nan, .2], [1, 1, 1], config)
        self.assertEqual(triples, [])

    def test_actual_scorer_capture_preserves_logits_and_approval(self):
        anchor, pairs, priors, config = fixture()
        original_load, original_example = mitosis.load_head, mitosis.example_tensors
        fake = lambda images, geometry: torch.full((len(images),), 2.)
        tensors = lambda coords, source: (np.zeros((4, 2, 2, 2), np.float32), np.zeros(11, np.float32))
        source = SimpleNamespace(scale=np.ones(3))
        with patch.object(mitosis, "load_head", return_value=fake), patch.object(mitosis, "example_tensors", side_effect=tensors):
            plain = mitosis.score_divisions(anchor.coords, pairs, priors, source, {}, {}, config, device="cpu")
            with capture_scores(anchor) as scores:
                observed = mitosis.score_divisions(anchor.coords, pairs, priors, source, {}, {}, config, device="cpu")
            self.assertEqual(plain, observed)
            self.assertEqual(scores, [{"ids": (0, 1, 2), "score": plain[0][3]}])
        self.assertIs(mitosis.load_head, original_load)
        self.assertIs(mitosis.example_tensors, original_example)

    def test_capture_restores_on_error(self):
        anchor, _, _, _ = fixture()
        original = mitosis.load_head
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with capture_scores(anchor):
                raise RuntimeError("fixture")
        self.assertIs(mitosis.load_head, original)

    def test_exact_fork_missing_candidate_and_low_score_are_distinct(self):
        anchor, _, _, _ = fixture()
        gt = {10: {11, 12}}
        mapping = {10: 0, 11: 1, 12: 2}
        for scored, reason in (({}, "not_scored"), ({(0, 1, 2): .2}, "below_threshold"),
                               ({(0, 1, 2): .9}, "approved_before_competition")):
            self.assertEqual(classify_divisions(gt, mapping, anchor, scored, .8)[0]["reason"], reason)
        self.assertEqual(classify_divisions(gt, {10: 0, 11: 1}, anchor, {}, .8)[0]["reason"], "unmatched_endpoint")

    def test_exact_fork_topology_blocks_are_distinct(self):
        anchor, _, _, _ = fixture()
        gt, mapping = {10: {11, 12}}, {10: 0, 11: 1, 12: 2}
        anchor.edges = np.array([[0, 1], [0, 2]])
        self.assertEqual(classify_divisions(gt, mapping, anchor, {}, .8)[0]["reason"], "already_present")
        anchor.edges = np.empty((0, 2), dtype=np.int64)
        self.assertEqual(classify_divisions(gt, mapping, anchor, {}, .8)[0]["reason"], "reference_mother_not_single")

    def test_score_summary_empty_is_not_zero_confidence(self):
        self.assertIsNone(score_summary([], .8)["quantiles"])
        self.assertEqual(score_summary([.1, .8, .9], .8)["at_frozen_threshold"], 2)
        with self.assertRaises(ValueError):
            score_summary([np.nan], .8)

    def test_batch_reconstruction_is_deterministic_without_mutation(self):
        records = [dict(dataset="a", label=i % 2) for i in range(6)]
        original = copy.deepcopy(records)
        config = dict(seed=42, epochs=2, batch_size=2)
        a = training_batch_summary(records, config)
        self.assertEqual(a, training_batch_summary(records, config))
        self.assertEqual(records, original)
        self.assertEqual(sum(a[0]["positive_counts"]), 3)

    def test_reuse_rejects_failed_source_forged_manifest_and_scientific_drift(self):
        manifest = {"core/production.py": "fixture", "configs/nextgen.json": "fixed", "ops/biohub.py": "old"}
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        receipt = dict(task="prepare", status="completed", result=dict(ready_for_experiments=True),
                       code_hash=identity, code_manifest=manifest)
        current = {**manifest, "ops/biohub.py": "new", "core/mitosis_diagnostic.py": "new"}
        with patch("core.mitosis_diagnostic.SOURCE_HASH", identity):
            verify_diagnostic_source(receipt, current)
            for bad in ({**receipt, "status": "failed"}, {**receipt, "code_hash": "forged"},
                        {**receipt, "code_manifest": {**manifest, "core/production.py": "forged"}}):
                with self.assertRaises(ValueError):
                    verify_diagnostic_source(bad, current)
            with self.assertRaisesRegex(ValueError, "scientific code drift"):
                verify_diagnostic_source(receipt, {**current, "configs/nextgen.json": "changed"})
            with self.assertRaisesRegex(ValueError, "scientific code drift"):
                verify_diagnostic_source(receipt, {**current, "core/unreviewed.py": "new"})

    def test_diagnostic_requires_current_tests_and_cannot_feed_experiment(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("prepared", "tests"):
                (root / "runs" / name).mkdir(parents=True)
            source = dict(task="prepare", status="completed", code_hash="old")
            (root / "runs/prepared/receipt.json").write_text(json.dumps(source))
            tests = dict(task="tests", status="completed", code_hash="new", result=dict(tests_passed=True))
            (root / "runs/tests/receipt.json").write_text(json.dumps(tests))
            args = SimpleNamespace(artifact_root=folder, source_run="prepared", tests_run="tests",
                                   code_hash="new", task="diagnose-mitosis")
            with patch("core.mitosis_diagnostic.verify_diagnostic_source"):
                self.assertEqual(source_run(args), root / "runs/prepared")
                args.tests_run = None
                with self.assertRaisesRegex(ValueError, "tests-run"):
                    source_run(args)
            args.task = "experiment"
            with self.assertRaisesRegex(ValueError, "identical frozen code"):
                source_run(args)
            source.update(task="diagnose-mitosis", code_hash="new")
            (root / "runs/prepared/receipt.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "Expected source task prepare"):
                source_run(args)


class MitosisDiagnosticRuntimeTests(unittest.TestCase):
    def test_official_matching_preserves_unsorted_anchor_indices(self):
        from core.evaluation import make_graph
        from core.mitosis_diagnostic import match_annotation
        anchor = Lineage(np.array([[1, 1, 1, 10], [0, 1, 1, 20], [1, 1, 1, 30]], float),
                         np.array([[1, 0], [1, 2]], dtype=np.int64))
        gt = make_graph({0: [{"id": 10, "pos": [1, 1, 20]}],
                         1: [{"id": 20, "pos": [1, 1, 10]}, {"id": 30, "pos": [1, 1, 30]}]},
                         [(10, 20), (10, 30)])
        expected = {int(r["node_id"]): next(i for i, c in enumerate(anchor.coords)
            if tuple(c) == tuple(r[k] for k in ("t", "z", "y", "x")))
            for r in gt.node_attrs().iter_rows(named=True)}
        with patch("ops.workflow.load_graph", return_value=gt):
            mapping, children, parents, triples = match_annotation(anchor, "fixture.geff", [1, 1, 1])
        self.assertEqual(mapping, expected)
        self.assertEqual(len(triples), 1)
        mother = next(iter(triples))
        self.assertEqual(mapping[mother], 1)
        self.assertEqual(len(children[mother]), 2)
        self.assertEqual(set(parents.values()), {mother})


if __name__ == "__main__":
    unittest.main()
