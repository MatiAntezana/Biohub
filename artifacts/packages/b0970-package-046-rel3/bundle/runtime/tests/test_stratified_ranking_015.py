"""015 toy fixture tests for stratified ranking; no real detector inference or GT fitting."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)

import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import stratified_ranking_015 as w
from core.artifacts import code_manifest, code_hash, dump
from core.nextgen import repair_ranker as r, event_ranker as p


def toy_data():
    x = np.zeros((6, len(r.NAMES)))
    x[0, 0] = 1.0   # beneficial 1
    x[1, 0] = 1.5   # beneficial 2
    x[2, 0] = -1.0  # harmful 1
    x[3, 0] = -2.0  # harmful 2
    x[4, 0] = 2.0   # event 1
    x[5, 0] = 2.5   # event 2
    return dict(
        x=x,
        y=np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0]),
        keys=np.array(["v1:1", "v2:1", "v1:2", "v2:2", "v1:3", "v2:3"]),
        strata=np.array(["beneficial", "beneficial", "harmful", "harmful", "event", "event"]),
    )


class StratifiedAssemblyTests(unittest.TestCase):
    def test_training_summary_counts_events_and_volumes(self):
        data = toy_data()
        summary = w.training_summary(data)
        self.assertEqual(summary["beneficial"]["comparisons"], 2)
        self.assertEqual(summary["beneficial"]["events"], 2)
        self.assertEqual(summary["beneficial"]["volumes"], 2)
        self.assertEqual(summary["harmful"]["comparisons"], 2)
        self.assertEqual(summary["event"]["comparisons"], 2)
        self.assertEqual(summary["unique_mothers_across_strata"], 6)

    def test_sufficient_requires_all_minimums(self):
        cfg = dict(
            minimum_beneficial_events=3,
            minimum_harmful_events=10,
            minimum_fork_events=3,
            minimum_fork_volumes=2,
        )
        self.assertFalse(w.sufficient(w.training_summary(toy_data()), cfg))

        mock_summary = dict(
            beneficial=dict(comparisons=5, events=4, volumes=4),
            harmful=dict(comparisons=728, events=713, volumes=12),
            event=dict(comparisons=5, events=3, volumes=3),
        )
        self.assertTrue(w.sufficient(mock_summary, cfg))

        mock_summary["event"]["events"] = 2
        self.assertFalse(w.sufficient(mock_summary, cfg))

    def test_event_weights_balance_strata_and_events(self):
        cfg = w.config()
        data = toy_data()
        weights = w.event_weights(data["keys"], data["strata"], cfg)
        self.assertEqual(len(weights), 6)
        self.assertAlmostEqual(weights[data["strata"] == "beneficial"].sum(), 0.25)
        self.assertAlmostEqual(weights[data["strata"] == "harmful"].sum(), 0.25)
        self.assertAlmostEqual(weights[data["strata"] == "event"].sum(), 0.50)
        self.assertAlmostEqual(weights.sum(), 1.0)


class StratifiedFittingTests(unittest.TestCase):
    def test_fit_ranker_converges_with_zero_intercept(self):
        cfg = w.config()
        data = toy_data()
        res = w.fit_event_ranker(data, cfg, variant="variant1")
        self.assertEqual(res["feature_names"], list(r.NAMES))
        self.assertEqual(res["intercept"], 0.0)
        self.assertEqual(len(res["weights"]), len(r.NAMES))
        self.assertLessEqual(res["fit"]["gradient_inf"], cfg["max_gradient_inf"])
        self.assertAlmostEqual(res["fit"]["stratum_mass"]["beneficial"], 0.25)
        self.assertAlmostEqual(res["fit"]["stratum_mass"]["harmful"], 0.25)
        self.assertAlmostEqual(res["fit"]["stratum_mass"]["event"], 0.50)

    def test_preference_metrics_computes_macro_fractions(self):
        cfg = w.config()
        data = toy_data()
        model_fit = w.fit_event_ranker(data, cfg)
        model = dict(
            feature_names=list(r.NAMES),
            weights=model_fit["weights"],
            intercept=0.0,
            standardizer=model_fit["standardizer"],
        )
        metrics = w.preference_metrics(data, model)
        for kind in ("beneficial", "harmful", "event"):
            self.assertIn(kind, metrics)
            self.assertGreaterEqual(metrics[kind]["fraction"], 0.0)
            self.assertLessEqual(metrics[kind]["fraction"], 1.0)

    def test_check_model_rejects_schema_drift(self):
        cfg = w.config()
        corpus = dict(
            splits=dict(
                **{"44b6": dict(fit=["44b6_1"], calibration=["44b6_cal"])},
                **{"6bba": dict(fit=["6bba_1"], calibration=["6bba_cal"])},
            ),
            development=["44b6_dev"],
            datasets={"44b6_dev": dict(group="44b6")},
        )
        model = dict(
            schema="wrong_schema",
            scope=w.SCOPE,
            recipe=cfg,
            training_groups=["44b6"],
            training_datasets=["44b6_1"],
            selection_groups=[],
            selection_datasets=[],
            diagnostic_margin=None,
        )
        with self.assertRaises(ValueError):
            w.check_model(model, corpus, "44b6_dev")


class StratifiedWorkflowTests(unittest.TestCase):
    def test_parent_manifest_preserves_historical_code(self):
        archived = {"core/production.py": "frozen", "ops/biohub.py": "old"}
        cfg = {**w.config(), "source_release": hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest()}
        current = {**archived, **dict.fromkeys(w.ADDITIONS, "new"), "ops/biohub.py": "new"}
        with patch.object(w, "config", return_value=cfg):
            self.assertEqual(w.parent_manifest(current, archived), archived)
            for bad in ({**current, "core/production.py": "changed"}, {**current, "new.py": "extra"}):
                with self.assertRaises(ValueError):
                    w.parent_manifest(bad, archived)

    def test_wrong_source_ids_fail_before_io(self):
        with self.assertRaises(ValueError):
            w.verify_sources(Path("wrong_source_run"), {})

    def test_contrast_label_and_noop(self):
        # Good equals other -> NOOP
        good = dict(add=[(0, 1), (0, 2)], remove=[])
        other = dict(add=[(0, 1), (0, 2)], remove=[])
        self.assertEqual(w.contrast_label(good, other, lambda e: 1), w.NOOP)

        # Good adds true edges, other adds wrong edges -> 1 (accepted)
        other2 = dict(add=[(0, 3)], remove=[])
        truth = lambda e: 1 if e in [(0, 1), (0, 2)] else 0
        self.assertEqual(w.contrast_label(good, other2, truth), 1)

    def test_discrimination_gate_requires_no_regression(self):
        by_group_good = {
            "44b6": {
                "event015_at_zero": {"confirmed_only": {"auroc": 0.85}, "new_unique_top1": 2},
                "ranker012_at_zero": {"confirmed_only": {"auroc": 0.80}, "new_unique_top1": 1},
                "event015_at_source_margin": {
                    "unique_best_selected": {"beneficial": 1, "harmful": 0, "mixed": 0, "unknown": 0}
                },
            },
            "6bba": {
                "event015_at_zero": {"confirmed_only": {"auroc": 0.90}, "new_unique_top1": 3},
                "ranker012_at_zero": {"confirmed_only": {"auroc": 0.88}, "new_unique_top1": 2},
                "event015_at_source_margin": {
                    "unique_best_selected": {"beneficial": 2, "harmful": 0, "mixed": 0, "unknown": 0}
                },
            },
        }
        gate = w.discrimination_gate(by_group_good)
        self.assertTrue(gate["passed"])

        by_group_bad = copy.deepcopy(by_group_good)
        by_group_bad["44b6"]["event015_at_zero"]["confirmed_only"]["auroc"] = 0.75
        gate = w.discrimination_gate(by_group_bad)
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
