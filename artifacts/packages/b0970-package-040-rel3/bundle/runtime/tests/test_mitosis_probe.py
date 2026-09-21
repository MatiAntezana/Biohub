"""Representation ablations: no data leakage, model promotion or threshold tuning."""
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

from core.artifacts import digest
from core.mitosis_probe import (ADDITIONS, DRIVERS, SCOPE, VARIANTS, config,
    feature_matrix, standardize_training, transform, class_weights, objective,
    fit_probe, predict_probe, probe_metrics, verify_probe_source, probe_job,
    extract_features, load_probe, compare_development, validate_recipe)


class MitosisProbeTests(unittest.TestCase):
    def test_fixed_variants_expose_only_their_own_features(self):
        geometry, image = np.ones((3, 11)), np.full((3, 64), 2.)
        for name, width in zip(VARIANTS, (11, 64, 75)):
            x, names = feature_matrix(geometry, image, name)
            self.assertEqual(x.shape, (3, width))
            self.assertEqual(len(names), width)
        np.testing.assert_array_equal(feature_matrix(geometry, image, "geometry")[0], geometry)
        np.testing.assert_array_equal(feature_matrix(geometry, image, "frozen_image")[0], image)
        with self.assertRaises(ValueError):
            feature_matrix(geometry, image, "fourth_variant")
        with self.assertRaises(ValueError):
            feature_matrix(geometry, image[:, :63], "geometry")
        image[0, 0] = np.nan
        with self.assertRaises(ValueError):
            feature_matrix(geometry, image, "frozen_image")

    def test_standardization_uses_only_training_and_discards_constant_columns(self):
        train = np.array([[1., 7.], [3., 7.]])
        x, stats = standardize_training(train, 1e-6)
        before = copy.deepcopy(stats)
        np.testing.assert_allclose(x, [[-1, 0], [1, 0]])
        np.testing.assert_allclose(transform([[102, -999]], stats), [[100, 0]])
        self.assertEqual(stats, before)
        np.testing.assert_array_equal(train, [[1, 7], [3, 7]])
        with self.assertRaises(ValueError):
            transform([[1, np.inf]], stats)

    def test_class_weights_are_exact_half_per_class_not_double_balanced(self):
        labels = np.array([1, 1] + [0] * 9)
        weights = class_weights(labels)
        self.assertAlmostEqual(weights[labels == 1].sum(), .5)
        self.assertAlmostEqual(weights[labels == 0].sum(), .5)
        for bad in ([0, 0], [1, -1], [1, .5, 0]):
            with self.assertRaises(ValueError):
                class_weights(bad)

    def test_objective_gradient_and_unpenalized_intercept(self):
        x = np.array([[-1., 2.], [1., 3.], [2., -1.]])
        labels = np.array([0, 0, 1])
        weights, theta = class_weights(labels), np.array([.2, -.3, .7])
        value, grad = objective(theta, x, labels, weights, 1.)
        numeric = []
        for i in range(3):
            step = np.eye(3)[i] * 1e-6
            numeric.append((objective(theta + step, x, labels, weights, 1.)[0]
                            - objective(theta - step, x, labels, weights, 1.)[0]) / 2e-6)
        np.testing.assert_allclose(grad, numeric, atol=1e-8)
        unpenalized, grad0 = objective(theta, x, labels, weights, 0.)
        self.assertAlmostEqual(value - unpenalized, .5 * (theta[:-1] @ theta[:-1]))
        self.assertEqual(grad[-1], grad0[-1])

    def test_average_precision_is_tie_aware_and_not_probability_calibration(self):
        self.assertEqual(probe_metrics([0, 1, 0, 1], [3, 3, 3, 3], .8)["average_precision"], .5)
        self.assertEqual(probe_metrics([0, 1], [1000, 1001], .8)["average_precision"], 1.)
        self.assertAlmostEqual(probe_metrics([1, 0, 1], [3, 2, 1], .8)["average_precision"], 5/6)
        self.assertIsNone(probe_metrics([0], [1], .8)["average_precision"])
        self.assertIsNone(probe_metrics([], [], .8)["average_precision"])

    def test_recipe_rejects_ad_hoc_regularization_or_threshold_before_fit(self):
        for key, value in (("l2", .01), ("evaluation_threshold", .1), ("variants", ["geometry"]),
                           ("selection", "best_development")):
            with patch("core.mitosis_probe.minimize") as optimizer, self.assertRaises(ValueError):
                fit_probe([[-1], [1]], [0, 1], {**config(), key: value})
            optimizer.assert_not_called()

    def test_unconverged_or_large_gradient_never_produces_a_model(self):
        for success, theta in ((False, np.zeros(2)), (True, np.array([100., 100.]))):
            result = SimpleNamespace(success=success, x=theta, message="stopped", nit=1)
            with patch("core.mitosis_probe.minimize", return_value=result), self.assertRaisesRegex(ValueError, "did not converge"):
                fit_probe([[-1], [1]], [0, 1], config())

    def test_source_science_is_exact_except_two_drivers_and_three_additions(self):
        previous = {"core/nextgen/mitosis.py": "f"*64, "core/mitosis_learning.py": "a"*64}
        cfg = {**config(), "parent_science_sha256": hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()}
        current = {**previous, **dict.fromkeys(DRIVERS | ADDITIONS, "new")}
        with patch("core.mitosis_probe.config", return_value=cfg), patch("core.mitosis_probe.verify_learning_source") as verify:
            verify_probe_source(Path("b0970-prepare-005"), current)
            self.assertFalse(ADDITIONS & set(verify.call_args.args[1]))
            for bad in ({**current, "core/nextgen/mitosis.py": "changed"},
                        {**current, "core/unreviewed.py": "new"},
                        {p: s for p, s in current.items() if p != "core/mitosis_learning.py"}):
                with self.assertRaisesRegex(ValueError, "unchanged 007 science"):
                    verify_probe_source(Path("b0970-prepare-005"), bad)
            with self.assertRaisesRegex(ValueError, "pinned 005"):
                verify_probe_source(Path("b0970-learn-007"), current)

    def test_source_asset_corruption_is_not_ignored(self):
        cfg = {**config(), "parent_science_sha256": hashlib.sha256(b"{}").hexdigest()}
        with patch("core.mitosis_probe.config", return_value=cfg), patch("core.mitosis_probe.verify_learning_source", side_effect=ValueError("corrupt asset")):
            with self.assertRaisesRegex(ValueError, "corrupt asset"):
                verify_probe_source(Path("b0970-prepare-005"), dict.fromkeys(DRIVERS | ADDITIONS, "new"))

    def test_six_fits_finish_before_development_and_job_has_no_promotion(self):
        events = []
        def train(source, group, out, cfg):
            events.extend((group, variant) for variant in VARIANTS)
            return {v: dict(path=str(out / (v + ".json")), sha256="f"*64) for v in VARIANTS}
        def compare(*args):
            self.assertEqual(events, [(g, v) for g in ("44b6", "6bba") for v in VARIANTS])
            events.append("development")
            return {}
        with tempfile.TemporaryDirectory() as folder, patch("core.mitosis_probe.verify_probe_source") as verify, patch("core.mitosis_probe.train_group", side_effect=train), patch("core.mitosis_probe.compare_development", side_effect=compare), patch("torch.cuda.empty_cache"):
            result = probe_job(SimpleNamespace(), Path(folder), Path("b0970-prepare-005"))
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(result["disposition"], SCOPE)
            for key in ("tracking_evaluated", "confirmation_evaluated", "threshold_selected", "model_selected"):
                self.assertFalse(result[key])
            self.assertFalse((Path(folder) / "submission.csv").exists())

    def test_probe_loader_rejects_same_group_encoder_drift_or_production_schema(self):
        model = dict(schema="mitosis_linear_probe_v1", scope=SCOPE, variant="geometry",
            training_groups=["44b6"], selection_groups=[], recipe=config())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.json"
            path.write_text(json.dumps(model))
            spec = dict(path=str(path), sha256=digest(path))
            self.assertEqual(load_probe(spec, "6bba", "44b6", "geometry"), model)
            with self.assertRaises(ValueError):
                load_probe(spec, "44b6", "44b6", "geometry")
            for change in (dict(schema=1), dict(scope="production"), dict(selection_groups=["6bba"]),
                           dict(training_groups=["6bba"]), dict(variant="frozen_image", encoder=dict(sha256="f"*64))):
                changed = {**model, **change}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_probe(dict(path=str(path), sha256=digest(path)), "6bba", "44b6", changed["variant"])

    def test_development_uses_opposite_encoder_and_never_confirmation_rows(self):
        def cache(source, group):
            return [dict(dataset=group+s, label=y) for s, y in (("_dev", 1), ("_dev", 0), ("_confirmation", 1))], None, None
        def extract(source, group, other, indices):
            self.assertNotEqual(group, other)
            np.testing.assert_array_equal(indices, [0, 1])
            return cache(source, group)[0][:2], np.zeros((2, 11)), np.zeros((2, 64)), np.array([2., -2.])
        with tempfile.TemporaryDirectory() as folder, patch("core.mitosis_probe.PROTOCOL", dict(group_prefixes=["44b6", "6bba"], development=["44b6_dev", "6bba_dev"])), patch("core.mitosis_probe.open_cache", side_effect=cache), patch("core.mitosis_probe.extract_features", side_effect=extract), patch("core.mitosis_probe.load_probe", return_value={}) as loader, patch("core.mitosis_probe.predict_probe", return_value=np.array([2., -2.])):
            result = compare_development(Path("source"), {g: dict.fromkeys(VARIANTS, {}) for g in ("44b6", "6bba")}, config(), Path(folder))
            self.assertEqual(loader.call_count, 6)
            self.assertEqual(result["44b6"]["encoder_and_probe_trained_on_group"], "6bba")
            for group in result.values():
                self.assertEqual(set(group["by_variant"]), {"historical", *VARIANTS})
                for row in group["by_variant"].values():
                    self.assertEqual(row["metrics"]["examples"], 2)

    def test_probe_requires_new_tests_and_cannot_feed_experiment_or_package(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("source", "tests"):
                (root / "runs" / name).mkdir(parents=True)
            source = dict(status="completed", task="prepare", code_hash="old")
            checked = dict(status="completed", task="tests", code_hash="new", result=dict(tests_passed=True))
            (root / "runs/source/receipt.json").write_text(json.dumps(source))
            (root / "runs/tests/receipt.json").write_text(json.dumps(checked))
            args = SimpleNamespace(task="probe-mitosis", source_run="source", tests_run="tests", artifact_root=folder, code_hash="new")
            with patch("core.mitosis_probe.verify_probe_source"):
                self.assertEqual(source_run(args), root / "runs/source")
                checked["code_hash"] = "old"
                (root / "runs/tests/receipt.json").write_text(json.dumps(checked))
                with self.assertRaises(ValueError):
                    source_run(args)
                args.tests_run = None
                with self.assertRaises(ValueError):
                    source_run(args)
            source.update(task="probe-mitosis", code_hash="new")
            (root / "runs/source/receipt.json").write_text(json.dumps(source))
            for task in ("experiment", "package", "prepare"):
                args.task = task
                with self.assertRaisesRegex(ValueError, "Expected source task"):
                    source_run(args)


class MitosisProbeRuntimeTests(unittest.TestCase):
    def test_toy_end_to_end_writes_six_models_and_only_development_scores(self):
        def cache(source, group):
            rows = [dict(dataset=group+suffix, label=label) for suffix, label in
                (("_dev", 0), ("_dev", 1), ("_other", 0), ("_other", 1))]
            return rows, None, None
        def extract(source, group, encoder, indices):
            rows = [cache(source, group)[0][i] for i in indices]
            signal = np.array([2*r["label"]-1 for r in rows])
            geom, image = np.zeros((len(rows), 11)), np.zeros((len(rows), 64))
            geom[:, 0], image[:, 0] = signal, signal
            return rows, geom, image, signal
        with tempfile.TemporaryDirectory() as folder, patch("core.mitosis_probe.verify_probe_source"), patch("core.mitosis_probe.PROTOCOL", dict(group_prefixes=["44b6", "6bba"], development=["44b6_dev", "6bba_dev"])), patch("core.mitosis_probe.open_cache", side_effect=cache), patch("core.mitosis_probe.extract_features", side_effect=extract), patch("torch.cuda.empty_cache"):
            root = Path(folder)
            result = probe_job(SimpleNamespace(), root, Path("b0970-prepare-005"))
            self.assertEqual(result["probe_sha256"], digest(root / "probe.json"))
            self.assertEqual(len(list(root.glob("train-*/*.json"))), 6)
            report = json.loads((root / "probe.json").read_text())
            for group in ("44b6", "6bba"):
                scores = json.loads((root / (group + ".scores.json")).read_text())
                self.assertEqual(len(scores), 2)
                self.assertTrue(all(row["dataset"].endswith("_dev") for row in scores))
                for variant in report["development"][group]["by_variant"].values():
                    self.assertEqual(variant["metrics"]["average_precision"], 1.)
            self.assertFalse(report["model_selected"])

    def test_toy_convex_fit_is_finite_reproducible_and_serializable(self):
        # Remote-only optimizer smoke, not real-data learning on the Mac.
        x, y = np.array([[-2., 5.], [-1., 5.], [1., 5.], [2., 5.]]), np.array([0, 0, 1, 1])
        first, second = fit_probe(x, y, config()), fit_probe(x, y, config())
        self.assertEqual(first, second)
        recovered = json.loads(json.dumps(first, allow_nan=False))
        logits = predict_probe(recovered, x)
        self.assertEqual(probe_metrics(y, logits, .8)["auroc"], 1.)
        self.assertGreater(recovered["weights"][0], 0)
        self.assertEqual(recovered["weights"][1], 0)
        self.assertLessEqual(recovered["fit"]["gradient_inf"], config()["max_gradient_inf"])

    def test_feature_extraction_freezes_encoder_and_uses_exact_requested_indices(self):
        class ToyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.))
            def forward(self, image):
                return image.reshape(len(image), -1)[:, :1].repeat(1, 64) * self.scale
        head = torch.nn.Module()
        head.image, head.classifier = ToyEncoder(), torch.nn.Linear(75, 1)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        records = [dict(dataset="6bba_dev", label=i % 2) for i in range(3)]
        images, geometry = np.arange(3, dtype=np.float32).reshape(3, 1), np.zeros((3, 11), np.float32)
        with patch("core.mitosis_probe.open_cache", return_value=(records, images, geometry)), patch("core.mitosis_probe.verify_file"), patch("core.mitosis_probe.torch.load", return_value=dict(training_groups=["44b6"], selection_groups=[])), patch("core.mitosis_probe.load_head", return_value=head) as loader:
            rows, geom, encoded, logits = extract_features(Path("source"), "6bba", "44b6", np.array([2, 0]), device="cpu")
        np.testing.assert_array_equal(encoded[:, 0], [4, 0])
        self.assertEqual(rows, [records[2], records[0]])
        self.assertEqual(loader.call_args.args[1], dict(group="6bba", split="confirmation"))
        self.assertTrue(np.isfinite(logits).all())
        for name, value in head.state_dict().items():
            torch.testing.assert_close(value, before[name])
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.parameters()))


if __name__ == "__main__":
    unittest.main()
