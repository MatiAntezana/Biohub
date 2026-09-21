"""Regression tests for the observed failure modes and delegation boundaries."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.artifacts import ROOT, dump
from core.nextgen.contracts import Lineage, check_manifest, load_candidates
from core.nextgen.mitosis import add_confirmed_pairs, train_head, load_head
from core.protocol import assess, panel_names, PROTOCOL
from core.remote_executor import DEFAULT_NODES, require_worker


def evaluation(delta):
    ref = {"score": .9, "edge_jaccard": .9, "division_jaccard": .4}
    cand = {**ref, "score": .9 + delta, "edge_jaccard": .9 + delta}
    return {"summary": {"reference": ref, "candidate": cand},
        "by_group": {g: {"reference": ref, "candidate": cand} for g in ("44b6", "6bba")},
        "rows": {"reference": [{"dataset": "x", "adj_edge_jaccard": .9}],
                 "candidate": [{"dataset": "x", "adj_edge_jaccard": .9 + delta}]}}


class ProtocolTests(unittest.TestCase):
    def test_confirmation_failure_blocks_even_after_strong_development(self):
        self.assertTrue(assess(evaluation(.01), "development")["passed"])
        self.assertFalse(assess(evaluation(-.0006208), "confirmation")["passed"])

    def test_group_regression_and_undefined_divisions_cannot_be_hidden_by_average(self):
        e = evaluation(.02)
        e["by_group"]["44b6"] = copy.deepcopy(e["by_group"]["44b6"])
        e["by_group"]["44b6"]["candidate"]["score"] = .89
        self.assertFalse(assess(e, "development")["passed"])
        e = evaluation(.02)
        e["summary"]["candidate"]["division_jaccard"] = float("nan")
        self.assertFalse(assess(e, "development")["passed"])

    def test_confirmation_selection_excludes_every_old_panel(self):
        names = set(PROTOCOL["development"] + PROTOCOL["previously_exposed"])
        names.update(f"{g}_new{i:02d}" for g in PROTOCOL["group_prefixes"] for i in range(12))
        info = {n: {"divisions": i % 3, "estimated_nodes": i * 100 + 1} for i, n in enumerate(sorted(names))}
        a = panel_names(names, info)
        b = panel_names(reversed(sorted(names)), info)
        self.assertEqual(a, b)
        self.assertEqual(len(a["confirmation"]), 8)
        self.assertFalse(set(a["confirmation"]) & set(PROTOCOL["previously_exposed"] + PROTOCOL["development"]))

    def test_mitosis_add_preserves_edges_and_resolves_competing_parents(self):
        coords = np.array([[0, 1, 1, 1], [0, 1, 1, 5], [1, 1, 1, 1], [1, 1, 1, 5], [1, 1, 1, 3]], float)
        anchor = Lineage(coords, np.array([[0, 2], [1, 3]], dtype=np.int64))
        graph, report = add_confirmed_pairs(anchor, [(0, 2, 4, .90), (1, 3, 4, .95)])
        self.assertEqual(set(map(tuple, graph.edges)), {(0, 2), (1, 3), (1, 4)})
        np.testing.assert_array_equal(graph.coords, anchor.coords)
        self.assertEqual(report["preserved_reference_edges"], 2)

    def test_mitosis_cannot_steal_a_child_or_replace_a_division(self):
        coords = np.array([[0, 1, 1, 1], [0, 1, 1, 5], [1, 1, 1, 1], [1, 1, 1, 5], [1, 1, 1, 3]], float)
        anchor = Lineage(coords, np.array([[0, 2], [0, 4], [1, 3]], dtype=np.int64))
        graph, _ = add_confirmed_pairs(anchor, [(1, 3, 4, .99), (0, 2, 3, .999)])
        self.assertEqual(set(map(tuple, graph.edges)), set(map(tuple, anchor.edges)))

    def test_fixed_epoch_training_rejects_opposite_group_before_images_are_opened(self):
        config = json.loads((ROOT / "configs/nextgen.json").read_text())["training"]
        rows = [{"name": "a", "group": "44b6", "split": "train"}, {"name": "b", "group": "6bba", "split": "validation"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text(json.dumps({"dataset": "b"}) + "\n")
            with self.assertRaisesRegex(ValueError, "must not read an evaluation group"):
                train_head(path, {"datasets": rows}, Path(folder)/"never.pt", config)

    def test_confirmation_head_cannot_have_seen_its_group(self):
        import torch
        with patch("core.nextgen.mitosis.verify_file"), patch.object(torch, "load", return_value={
            "schema_version": 1, "feature_dim": 11, "training_groups": ["44b6"], "selection_groups": ["6bba"]}):
            with self.assertRaisesRegex(ValueError, "used to train or select"):
                load_head({"checkpoint": "fixture", "sha256": "fixture"}, {"split": "confirmation", "group": "6bba"}, "cpu")

    def test_mac_cannot_impersonate_remote_worker(self):
        with patch("core.remote_executor.platform.system", return_value="Darwin"):
            with self.assertRaises(RuntimeError):
                require_worker(DEFAULT_NODES["A1688"])

    def test_submission_scope_prohibits_ground_truth(self):
        m = {"schema_version": 1, "scope": "submission", "datasets": [{"name": "sample", "group": "g",
            "split": "test", "scale_um": [1.625, .40625, .40625], "gt": "must-not-be-read"}]}
        with self.assertRaisesRegex(ValueError, "no GT"):
            check_manifest(m)

    def test_restored_v19_has_real_effective_scale_and_no_active_cycle_claim(self):
        nb = json.loads((ROOT / "legacy/v19_restored.ipynb").read_text())
        source = "".join(nb["cells"][5]["source"])
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_position_um")
        self.assertIn("EFFECTIVE_SCALE_Z", ast.unparse(fn))
        self.assertIn('BIOHUB_CYCLE_CONSISTENCY_ENABLE"] = "0"', "".join(nb["cells"][0]["source"]))

    def test_atomic_receipt_undefined_is_null(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"receipt.json"
            dump(path, {"score": float("nan"), "count": np.int64(7)})
            self.assertEqual(json.loads(path.read_text()), {"score": None, "count": 7})

    def test_kaggle_download_requires_exact_complete_without_failure(self):
        from ops.kaggle import require_complete_kernel, status_name
        from enum import Enum
        State = Enum("State", ["COMPLETE", "RUNNING", "INCOMPLETE"])
        for value in (State.RUNNING, State.INCOMPLETE, "ERROR", "KernelWorkerStatus.QUEUED"):
            response = SimpleNamespace(status=value, failure_message="")
            with self.subTest(status=value), self.assertRaises(RuntimeError):
                require_complete_kernel(SimpleNamespace(kernels_status=lambda _: response), "owner/run")
        response = SimpleNamespace(status=State.COMPLETE, failure_message="")
        self.assertIs(require_complete_kernel(SimpleNamespace(kernels_status=lambda _: response), "owner/run"), response)
        response.failure_message = "failed output finalization"
        with self.assertRaises(RuntimeError):
            require_complete_kernel(SimpleNamespace(kernels_status=lambda _: response), "owner/run")
        self.assertNotEqual(status_name("SubmissionStatus.INCOMPLETE"), "COMPLETE")

    def test_kaggle_publish_rejects_modified_archive(self):
        from core.artifacts import digest
        from ops.kaggle import ready
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            fields = {"kaggle/submission.ipynb": "notebook_sha256", "bundle/bundle.json": "bundle_sha256",
                      "dataset-upload/biohub_payload.zip": "archive_sha256"}
            meta = {"gates": {phase: {"passed": True} for phase in ("development", "confirmation")}}
            for relative, key in fields.items():
                target = package / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture")
                meta[key] = digest(target)
            dump(package / "publish-ready.json", meta)
            self.assertEqual(ready(package), meta)
            (package / "dataset-upload/biohub_payload.zip").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "modified after validation"):
                ready(package)


if __name__ == "__main__":
    unittest.main()
