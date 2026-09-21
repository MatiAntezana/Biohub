"""Fixed balanced batches, paired metrics and non-publishable learning assays."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.mitosis_learning import (balanced_batches, classification_metrics,
    development_indices, compare_development, config, learning_job, train_balanced,
    verify_learning_source)


class MitosisLearningTests(unittest.TestCase):
    def test_every_batch_has_exact_class_balance_and_is_reproducible(self):
        labels = np.array([1, 1] + [0] * 37)
        a = list(balanced_batches(labels, 16, 5, np.random.default_rng(42)))
        b = list(balanced_batches(labels, 16, 5, np.random.default_rng(42)))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(len(x), 16)
            self.assertEqual(labels[x].sum(), 8)
        self.assertEqual(set(np.concatenate(a)), set(range(len(labels))))

    def test_sampling_rejects_unknown_or_missing_classes(self):
        for labels, size in (([0, 0], 16), ([0, 1, -1], 16), ([0, 1], 3)):
            with self.assertRaises(ValueError):
                list(balanced_batches(labels, size, 1, np.random.default_rng(42)))

    def test_metrics_handle_ties_and_single_class_without_fake_auc(self):
        tied = classification_metrics([1, 0, 1, 0], [0, 0, 0, 0], .8)
        self.assertEqual(tied["auroc"], .5)
        self.assertAlmostEqual(tied["balanced_bce"], np.log(2))
        perfect = classification_metrics([0, 0, 1, 1], [-100, -2, 2, 100], .8)
        self.assertEqual(perfect["auroc"], 1)
        self.assertEqual((perfect["tp"], perfect["fp"]), (2, 0))
        self.assertIsNone(classification_metrics([1], [2], .8)["auroc"])
        self.assertIsNone(classification_metrics([], [], .8)["positive_recall"])
        with self.assertRaises(ValueError):
            classification_metrics([0], [np.nan], .8)
        with self.assertRaises(ValueError):
            classification_metrics([0.5], [0], .8)
        self.assertEqual(classification_metrics([0, 1], [1000, 1001], .8)["auroc"], 1)

    def test_development_selector_never_includes_other_panels(self):
        records = [dict(dataset=n) for n in ("development", "confirmation", "visible", "training")]
        np.testing.assert_array_equal(development_indices(records, {"development"}), [0])

    def test_double_balancing_and_epoch_selection_are_rejected_before_data(self):
        cfg = config()
        for field, value in (("positive_loss_weight", 89.58), ("selection", "best_development_epoch")):
            with self.subTest(field=field), patch("core.mitosis_learning.open_cache") as opened, self.assertRaises(ValueError):
                train_balanced(Path("unused"), "44b6", Path("unused"), {**cfg, field: value})
            opened.assert_not_called()

    def test_assay_checks_pinned_assets_before_any_training(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "b0970-prepare-005"
            source.mkdir()
            with patch("core.mitosis_learning.verify_file", side_effect=ValueError("corrupt asset")):
                with self.assertRaisesRegex(ValueError, "corrupt asset"):
                    verify_learning_source(source, {})

    def test_both_fixed_heads_finish_before_development_is_read(self):
        events = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def train(source, group, out, cfg):
                events.append(group)
                return dict(checkpoint=str(out / "head.pt"), sha256="f" * 64)
            def compare(*args):
                self.assertEqual(events, ["44b6", "6bba"])
                events.append("development")
                return {}
            with patch("core.mitosis_learning.verify_learning_source"), patch("core.mitosis_learning.train_balanced", side_effect=train), patch("core.mitosis_learning.compare_development", side_effect=compare), patch("torch.cuda.empty_cache"):
                result = learning_job(SimpleNamespace(), root, Path("b0970-prepare-005"))
            self.assertEqual(result["disposition"], "learning_assay_only_no_submission")
            self.assertFalse(result["tracking_evaluated"])
            self.assertFalse(result["confirmation_evaluated"])
            self.assertFalse((root / "submission.csv").exists())

    def test_comparison_names_actual_training_group_and_excludes_confirmation(self):
        cfg = config()
        models = {g: dict(checkpoint=g, sha256="f"*64) for g in ("44b6", "6bba")}
        names = {"44b6_dev", "6bba_dev"}
        def cache(source, group):
            records = [dict(dataset=group+suffix, label=label) for suffix, label in (("_dev", 1), ("_dev", 0), ("_confirmation", 1))]
            return records, np.zeros((3, 1)), np.zeros((3, 1))
        def predict(head, images, geom, indices, *args):
            np.testing.assert_array_equal(indices, [0, 1])
            return np.array([2., -2.])
        with tempfile.TemporaryDirectory() as folder, patch("core.mitosis_learning.PROTOCOL", dict(development=list(names), group_prefixes=["44b6", "6bba"])), patch("core.mitosis_learning.open_cache", side_effect=cache), patch("core.mitosis_learning.predict_cached", side_effect=predict), patch("core.mitosis_learning.load_head", return_value=object()) as loader:
            result = compare_development(Path("source"), models, cfg, Path(folder), device="cpu")
            self.assertEqual(result["44b6"]["head_trained_on_group"], "6bba")
            self.assertEqual(result["6bba"]["head_trained_on_group"], "44b6")
            self.assertEqual(loader.call_count, 4)
            for call in loader.call_args_list:
                self.assertEqual(call.args[1]["split"], "confirmation")

    def test_learning_requires_current_tests_and_cannot_feed_package(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("source", "tests"):
                (root / "runs" / name).mkdir(parents=True)
            source = dict(status="completed", task="prepare", code_hash="old")
            (root / "runs/source/receipt.json").write_text(json.dumps(source))
            checked = dict(status="completed", task="tests", code_hash="new", result=dict(tests_passed=True))
            (root / "runs/tests/receipt.json").write_text(json.dumps(checked))
            args = SimpleNamespace(source_run="source", tests_run="tests", artifact_root=folder, code_hash="new", task="learn-mitosis")
            with patch("core.mitosis_learning.verify_learning_source"):
                self.assertEqual(source_run(args), root / "runs/source")
                checked["code_hash"] = "old"
                (root / "runs/tests/receipt.json").write_text(json.dumps(checked))
                with self.assertRaises(ValueError):
                    source_run(args)
            source.update(task="learn-mitosis", code_hash="new")
            (root / "runs/source/receipt.json").write_text(json.dumps(source))
            args.task = "package"
            with self.assertRaisesRegex(ValueError, "Expected source task experiment"):
                source_run(args)


class MitosisLearningRuntimeTests(unittest.TestCase):
    def test_tiny_training_loop_saves_only_final_checkpoint_and_keeps_source(self):
        # Remote-only toy optimizer smoke; no real training in local ops/verify.py.
        class TinyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, image, geometry):
                return self.weight * geometry[:, 0]
        cfg = {**config(), "epochs": 2, "batch_size": 4, "positive_per_batch": 2}
        records = [dict(label=i % 2) for i in range(4)]
        images, geom = np.zeros((4, 1), np.float32), np.array([[-1], [1], [-1], [1]], np.float32)
        saved = geom.copy()
        with tempfile.TemporaryDirectory() as folder, patch("core.mitosis_learning.open_cache", return_value=(records, images, geom)), patch("core.mitosis_learning.MitosisHead", TinyHead), patch("core.mitosis_learning.digest", return_value="f" * 64):
            spec = train_balanced(Path("source"), "44b6", Path(folder), cfg, device="cpu")
            payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
            self.assertGreater(payload["state_dict"]["weight"].item(), 0)
            self.assertEqual(payload["training_groups"], ["44b6"])
            self.assertEqual(payload["selection_groups"], [])
            self.assertEqual(payload["class_balance"]["positive_loss_weight"], 1)
            info = json.loads((Path(folder) / "training.json").read_text())
            self.assertEqual(len(info["history"]), 2)
            self.assertTrue(all(r["all_negative_batches"] == 0 for r in info["history"]))
            np.testing.assert_array_equal(saved, geom)


if __name__ == "__main__":
    unittest.main()
