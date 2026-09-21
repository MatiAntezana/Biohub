"""011: pure prediction, partial-label training, split isolation and real dispatch."""
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.artifacts import dump, digest
from core.nextgen.contracts import Lineage
from core.nextgen import association as a
from core import mitosis_selector as w


def graph():
    return Lineage(np.array([[0, 8, 8, 8], [0, 8, 8, 14],
        [1, 8, 8, 7], [1, 8, 8, 9], [1, 8, 8, 15],
        [2, 8, 8, 6], [2, 8, 8, 10], [2, 8, 8, 16]], float),
        np.array([[0, 2], [1, 3], [2, 5], [3, 6], [4, 7]], np.int64)).validate()


def ctx(g=None):
    g = graph() if g is None else g
    # Distinct real-valued unit fixture descriptors, not production evidence.
    p = np.random.default_rng(42).normal(size=(len(g.coords), 27)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return a.context(g, [1, 1, 1], p, np.tile([.2, 1, 1], (len(g.coords), 1)))


def scores(c):
    es, fs = a.candidate_tables(c["anchor"], c["scale"], w.config())
    return dict.fromkeys(map(tuple, es), 0.), dict.fromkeys(map(tuple, fs), 0.)


def fake_evaluation(delta):
    ref = dict(score=.9, edge_jaccard=.85, adj_edge_jaccard=.85, division_jaccard=.5)
    cand = {**ref, "score": .9+delta, "division_jaccard": .5+10*delta}
    return dict(summary=dict(reference=ref, candidate=cand),
        by_group={"44b6": dict(reference=ref, candidate=cand)},
        rows=dict(reference=[dict(dataset="44b6_cal", **ref)], candidate=[dict(dataset="44b6_cal", **cand)]))


class ToyImage:
    shape = (3, 40, 40, 40)
    scale = np.ones(3)
    array = SimpleNamespace(store="toy")

    def patch(self, t, centre, half_um, shape):
        z, y, x = np.meshgrid(*[np.linspace(-half_um, half_um, n)+v
            for n, v in zip(shape, centre)], indexing="ij")
        return (np.sin(x*.7+t)+np.cos(y*.3)+z*.1).astype(np.float32)


class AssociationSelectorTests(unittest.TestCase):
    def test_candidates_keep_all_reference_edges_and_forks_outside_radius(self):
        g = graph()
        g.edges = np.array([[0, 2], [0, 4]], np.int64)
        cfg = {**w.config(), "radius_um": .1}
        edges, forks = a.candidate_tables(g, [1, 1, 1], cfg)
        self.assertEqual(set(map(tuple, edges)), set(map(tuple, g.edges)))
        self.assertEqual(forks.tolist(), [[0, 2, 4]])

    def test_enumeration_includes_occupied_daughters_and_is_deterministic(self):
        edges, forks = a.candidate_tables(graph(), [1, 1, 1], w.config())
        self.assertIn((0, 2, 3), set(map(tuple, forks)))
        self.assertIn((0, 3), set(map(tuple, edges)))
        np.testing.assert_array_equal(forks, a.candidate_tables(graph(), [1, 1, 1], w.config())[1])
        with self.assertRaisesRegex(ValueError, "budget"):
            a.candidate_tables(graph(), [1, 1, 1], {**w.config(), "max_pairs_per_dataset": 1})

    def test_features_symmetric_under_daughter_exchange(self):
        c = ctx()
        np.testing.assert_allclose(a.fork_features([[0, 2, 3]], c), a.fork_features([[0, 3, 2]], c))
        self.assertEqual(a.edge_features([[0, 2]], c).shape, (1, len(a.EDGE_NAMES)))
        self.assertEqual(a.fork_features([[0, 2, 3]], c).shape, (1, len(a.FORK_NAMES)))

    def test_daughter_image_identity_changes_features(self):
        c = ctx()
        before = a.fork_features([[0, 2, 3]], c)
        c["descriptors"][3] = c["descriptors"][0]
        after = a.fork_features([[0, 2, 3]], c)
        self.assertFalse(np.allclose(before, after))

    def test_patch_at_each_node_own_position_and_time(self):
        image = ToyImage()
        with patch.object(image, "patch", wraps=image.patch) as calls:
            p, stats = a.image_descriptors(graph(), image, w.config())
        self.assertEqual(calls.call_count, len(graph().coords))
        for call, row in zip(calls.call_args_list, graph().coords):
            self.assertEqual(call.args[0], int(row[0]))
            np.testing.assert_array_equal(call.args[1], row[1:])
        self.assertFalse(np.allclose(p[2], p[3]))
        self.assertTrue(np.isfinite(stats).all())

    def test_flat_and_boundary_patches_marked_explicitly(self):
        g = Lineage(np.array([[0, 0, 0, 0]], float), np.empty((0, 2), np.int64))
        image = ToyImage()
        with patch.object(image, "patch", side_effect=lambda *args, **kw: np.zeros(kw["shape"], np.float32)):
            p, stats = a.image_descriptors(g, image, w.config())
        self.assertEqual(stats[0, 1], 0)
        self.assertLess(stats[0, 2], 1)
        self.assertTrue((p == 0).all())

    def test_context_handles_missing_motion_without_fabricated_velocity(self):
        c = ctx()
        self.assertEqual(c["prev_valid"][0], 0)
        self.assertEqual(c["next_valid"][5], 0)
        f = a.edge_features([[0, 2]], c)[0]
        self.assertEqual(f[a.EDGE_NAMES.index("mother_velocity_known")], 0)
        self.assertEqual(f[a.EDGE_NAMES.index("forward_residual")], 0)

    def test_gain_pays_displaced_edge_and_orphan(self):
        c = ctx()
        es, fs = scores(c)
        es[(0, 3)], es[(1, 3)], fs[(0, 2, 3)] = 4., 3., 2.
        tx = a.transaction(0, [2, 3], c, es, fs, w.config())
        self.assertAlmostEqual(tx["gain"], 4-3+2-2)
        self.assertEqual(tx["remove"], [(1, 3)])
        self.assertEqual(tx["add"], [(0, 3)])

    def test_single_child_can_remove_bad_existing_fork(self):
        g = graph()
        g.edges = np.array([[0, 2], [0, 3], [2, 5], [3, 6]], np.int64)
        c = ctx(g)
        es, fs = scores(c)
        es[(0, 3)], fs[(0, 2, 3)] = -2., -3.
        tx = a.transaction(0, [2], c, es, fs, w.config())
        self.assertEqual(tx["gain"], 5.)
        self.assertEqual(tx["add"], [])
        cfg = {**w.config(), "max_edit_fraction": 1.}
        candidate, audit = a.apply_transactions(g, [tx], 1., cfg)
        self.assertNotIn((0, 3), set(map(tuple, candidate.edges)))
        self.assertEqual(audit["status"], "applied")

    def test_reference_zero_gain_and_tied_alternatives_abstain(self):
        c = ctx()
        es, fs = scores(c)
        self.assertEqual(a.transaction(0, [2], c, es, fs, w.config())["gain"], 0.)
        # Equal positive options for mother0 should not be broken by daughter ID.
        fake = {k: 0. for k in fs}
        fake[(0, 2, 3)], fake[(0, 2, 4)] = 10., 8.  # first displaces an orphan: -2
        proposals = a.propose_transactions(c, es, fake, w.config())
        self.assertFalse(any(p["mother"] == 0 for p in proposals))

    def test_transactions_conflict_deterministically_and_preserve_input(self):
        g, c = graph(), ctx()
        es, fs = scores(c)
        fs[(0, 2, 3)], fs[(1, 3, 4)] = 10., 5.
        tx = [a.transaction(0, [2, 3], c, es, fs, w.config()),
              a.transaction(1, [3, 4], c, es, fs, w.config())]
        cfg = {**w.config(), "max_edit_fraction": 1.}
        result, audit = a.apply_transactions(g, tx, 1., cfg)
        reversed_result, _ = a.apply_transactions(g, tx[::-1], 1., cfg)
        self.assertEqual(len(audit["selected"]), 1)
        np.testing.assert_array_equal(result.edges, reversed_result.edges)
        np.testing.assert_array_equal(g.edges, graph().edges)
        np.testing.assert_array_equal(result.coords, g.coords)
        self.assertFalse(np.shares_memory(result.coords, g.coords))

    def test_abstention_and_over_budget_return_unchanged_reference(self):
        g = graph()
        # Two disjoint, valid edge removal transactions; cap is one for this fixture.
        proposals = [dict(mother=m, daughters=[d], gain=5., add=[], remove=[(m, d)],
            affected_parents=[m], touched_children=[d]) for m, d in [(2, 5), (3, 6)]]
        for margin, status in [(None, "abstained"), (1., "rejected_edit_budget")]:
            result, audit = a.apply_transactions(g, proposals, margin, w.config())
            np.testing.assert_array_equal(g.edges, result.edges)
            self.assertEqual(audit["status"], status)

    def test_nonfinite_scores_and_margins_fail_closed(self):
        for bad in (float("nan"), float("inf"), -1.):
            with self.assertRaises(ValueError):
                a.apply_transactions(graph(), [], bad, w.config())
        c = ctx()
        es, fs = scores(c)
        fs[(0, 2, 3)] = float("nan")
        with self.assertRaises(ValueError):
            a.transaction(0, [2, 3], c, es, fs, w.config())

    def test_invalid_transaction_topology_rejected(self):
        c = ctx()
        es, fs = scores(c)
        for ds in ([2, 2], [1], [], [2, 3, 4], [-1], [99], [2.2]):
            with self.assertRaises(ValueError):
                a.transaction(0, ds, c, es, fs, w.config())

    def test_predictor_has_no_annotation_or_oracle_interface(self):
        source = inspect.getsource(a)
        for forbidden in ("match_annotation", "oracle_copy", "gt_path", "read_text", "geff", "confirmed_label"):
            self.assertNotIn(forbidden, source)
        self.assertEqual(list(inspect.signature(a.apply_transactions).parameters), ["anchor", "proposals", "margin", "cfg"])

    def test_pure_linear_schema_requires_finite_training_standardizer(self):
        model = dict(weights=[1., 2.], intercept=0., standardizer=dict(mean=[0., 1.], scale=[1., 2.], active=[True, False]))
        np.testing.assert_allclose(a.linear_scores(model, np.array([[2., 1000.]])), [2.])
        model["standardizer"]["scale"][0] = 0.
        with self.assertRaises(ValueError):
            a.linear_scores(model, np.ones((1, 2)))

    def test_scoring_in_batches_equals_direct_frozen_features(self):
        c = ctx()
        edges, forks = a.candidate_tables(graph(), [1, 1, 1], w.config())
        def model(names):
            n = len(names)
            return dict(feature_names=list(names), weights=np.arange(n).tolist(), intercept=.1,
                standardizer=dict(mean=[0.]*n, scale=[1.]*n, active=[True]*n))
        m = dict(edge=model(a.EDGE_NAMES), fork=model(a.FORK_NAMES))
        es, fs = a.score_tables(edges, forks, c, m, {**w.config(), "feature_batch_size": 1})
        np.testing.assert_allclose(list(es.values()), a.linear_scores(m["edge"], a.edge_features(edges, c)))
        np.testing.assert_allclose(list(fs.values()), a.linear_scores(m["fork"], a.fork_features(forks, c)))


class SelectorWorkflowTests(unittest.TestCase):
    def test_splits_are_disjoint_exclude_all_reserved_and_use_no_predictions(self):
        cfg = w.config()
        names = [f"{g}_{i:03d}" for g in ("44b6", "6bba") for i in range(20)]
        plan = dict(all_datasets=names, panels=dict(development=w.PROTOCOL["development"],
            confirmation=["44b6_019", "6bba_019"], visible=["44b6_018", "6bba_018"]),
            public_info={n: dict(divisions=int(int(n[-3:]) < 12)) for n in names})
        splits = w.split_corpus(plan, cfg)
        all_used = []
        for g, split in splits.items():
            self.assertEqual(len(split["fit"]), 11)
            self.assertEqual(len(split["calibration"]), 5)
            used = split["fit"]+split["calibration"]
            self.assertTrue(all(n.startswith(g) for n in used))
            self.assertEqual(len(used), len(set(used)))
            all_used += used
        self.assertFalse(set(all_used) & set(plan["panels"]["confirmation"]+plan["panels"]["visible"]))
        self.assertEqual(splits, w.split_corpus({**plan, "all_datasets": names[::-1]}, cfg))
        with self.assertRaises(ValueError):
            w.split_corpus({**plan, "all_datasets": names[:3]}, cfg)

    def test_partial_annotation_unknown_never_negative(self):
        mapping, parents = {10: 0, 12: 2, 13: 3}, {12: 10, 13: 11}
        self.assertEqual(w.edge_labels([[0, 2], [0, 3], [1, 2], [0, 4]], mapping, parents).tolist(), [1, 0, -1, -1])
        self.assertEqual(w.label_pairs([[0, 2, 3], [0, 2, 4]], mapping, parents).tolist(), [0, -1])
        with self.assertRaises(ValueError):
            w.edge_labels([[0, 2]], {10: 0, 11: 0}, parents)

    def test_fit_is_small_convex_and_train_only_normalized(self):
        x = np.array([[i, 1.] for i in range(16)])
        y = np.array([0]*12+[1]*4)
        model = w.fit_binary(x, y, ["value", "constant"], w.config())
        self.assertTrue(model["fit"]["success"])
        self.assertFalse(model["calibrated_probability"])
        self.assertEqual(model["standardizer"]["mean"], [7.5, 1.])
        self.assertEqual(model["standardizer"]["active"], [True, False])
        self.assertLess(a.linear_scores(model, x[[0]])[0], a.linear_scores(model, x[[-1]])[0])
        mean = model["standardizer"]["mean"].copy()
        a.linear_scores(model, np.array([[10000., -10000.]]))
        self.assertEqual(mean, model["standardizer"]["mean"])

    def test_fitting_unknowns_or_insufficient_data_is_error(self):
        for labels in ([0]*12+[-1]*4, [0]*14+[1]*2, [0]*16):
            with self.assertRaises(ValueError):
                w.fit_binary(np.ones((16, 1)), np.array(labels), ["x"], w.config())

    def test_cross_prefix_guard_includes_calibration(self):
        model = dict(schema="association_selector_v1_diagnostic", scope=w.SCOPE, recipe=w.config(),
            training_groups=["44b6"], selection_groups=["44b6"])
        w.check_model(model, "6bba")
        with self.assertRaisesRegex(ValueError, "prefix"):
            w.check_model(model, "44b6")
        with self.assertRaises(ValueError):
            w.check_model({**model, "selection_groups": ["6bba"]}, "6bba")
        w.check_model({**model, "selection_groups": []}, "44b6", calibration=True)
        with self.assertRaises(ValueError):
            w.check_model({**model, "selection_groups": []}, "6bba", calibration=True)

    def test_calibration_grid_is_fixed_and_ties_prefer_conservative_margin(self):
        cfg = w.config()
        results = {m: dict(evaluation=fake_evaluation(.005 if m in (2., 4.) else 0.), budget_rejected=False) for m in cfg["margin_grid"]}
        self.assertEqual(w.select_margin(results, cfg), 4.)
        results[4.]["budget_rejected"] = True
        self.assertEqual(w.select_margin(results, cfg), 2.)
        results[2.]["budget_rejected"] = True
        self.assertIsNone(w.select_margin(results, cfg))
        with self.assertRaises(ValueError):
            w.select_margin({1.: results[1.]}, cfg)

    def test_calibration_rejects_edge_division_and_sample_regression(self):
        cfg = w.config()
        for metric in ("edge_jaccard", "division_jaccard"):
            results = {m: dict(evaluation=fake_evaluation(.01), budget_rejected=False) for m in cfg["margin_grid"]}
            for r in results.values():
                r["evaluation"]["summary"]["candidate"][metric] = 0.
            self.assertIsNone(w.select_margin(results, cfg))
        results = {m: dict(evaluation=fake_evaluation(.01), budget_rejected=False) for m in cfg["margin_grid"]}
        for r in results.values():
            r["evaluation"]["rows"]["candidate"][0]["adj_edge_jaccard"] = .1
        self.assertIsNone(w.select_margin(results, cfg))

    def test_corpus_evidence_roundtrip_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"44b6_fixture.evidence.npz"
            info = w.save_evidence(p, graph(), ToyImage(), w.config())
            manifest = dict(datasets={"44b6_fixture": info})
            got, es, fs, c = w.open_evidence(Path(d), "44b6_fixture", manifest)
            np.testing.assert_array_equal(got.coords, graph().coords)
            self.assertTrue(len(es) and len(fs))
            self.assertEqual(c["stats"].shape, (8, 3))
            with self.assertRaises(ValueError):
                w.open_evidence(Path(d), "not_allowed", manifest)
            manifest["datasets"]["44b6_fixture"]["sha256"] = "0"*64
            with self.assertRaises(ValueError):
                w.open_evidence(Path(d), "44b6_fixture", manifest)

    def test_parent_identity_chain_rejects_unreviewed_science(self):
        cfg = {**w.config(), "parent_science_sha256": hashlib.sha256(b"{}").hexdigest()}
        manifest = dict.fromkeys(w.ADDITIONS | w.DRIVERS, "new")
        with patch.object(w, "config", return_value=cfg), patch.object(w, "verify_parent", return_value=({}, {})) as parent, patch.object(w, "verify_file"):
            w.verify_sources(Path(cfg["source_run"]), manifest)
            self.assertEqual(parent.call_args.args[1], dict.fromkeys(w.DRIVERS, "new"))
            with self.assertRaisesRegex(ValueError, "exact010"):
                w.verify_sources(Path(cfg["source_run"]), {**manifest, "core/other.py": "drift"})

    def test_resource_profile_keeps_detector_on_gpu_selector_on_limited_cpu(self):
        from ops.biohub import resource_profile, CPU_PROFILE, GPU_PROFILE, child_environment
        self.assertEqual(resource_profile("prepare-selector"), GPU_PROFILE)
        self.assertEqual(resource_profile("evaluate-selector"), CPU_PROFILE)
        env = child_environment("evaluate-selector")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["OMP_NUM_THREADS"], "2")

    def test_corpus_disk_guard_defers_before_consuming_run(self):
        from ops.biohub import resource_admission
        env = dict(disk_free_gib=20., gpu_processes="", gpu="RTX 4090, 24000 MiB, 0 %")
        self.assertTrue(resource_admission("prepare", env)["allowed"])
        result = resource_admission("prepare-selector", env)
        self.assertFalse(result["allowed"])
        self.assertIn("24 GiB", result["reasons"][0])
        env["disk_free_gib"] = 30.
        self.assertTrue(resource_admission("prepare-selector", env)["allowed"])

    def test_preparation_opens_no_gt_and_reuses_only_development(self):
        dev = ["44b6_dev", "6bba_dev"]
        splits = {g: dict(fit=[g+"_fit"], calibration=[g+"_cal"]) for g in ("44b6", "6bba")}
        train_names = sorted(n for g in splits.values() for ns in g.values() for n in ns)
        with tempfile.TemporaryDirectory() as d:
            source, run = Path(d)/"b0970-prepare-005", Path(d)/"new"
            run.mkdir()
            dump(source/"plan.json", dict(data_root="images"))
            def load(path, names):
                self.assertTrue(str(path).endswith("cache-training/reference.csv") or
                                str(path).endswith("cache-development/reference.csv"))
                return {n: graph() for n in names}
            with ExitStack() as stack:
                stack.enter_context(patch.object(w, "verify_sources"))
                stack.enter_context(patch.object(w, "split_corpus", return_value=splits))
                stack.enter_context(patch.object(w, "PROTOCOL", dict(development=dev)))
                stack.enter_context(patch.object(w, "load_reference", side_effect=load))
                cache = stack.enter_context(patch("ops.workflow.cache_panel"))
                stack.enter_context(patch("core.nextgen.images.ImageSource", return_value=ToyImage()))
                stack.enter_context(patch.object(w.shutil, "disk_usage", return_value=SimpleNamespace(free=100*1024**3)))
                gt = stack.enter_context(patch.object(w, "match_annotation", side_effect=AssertionError("GT forbidden during corpus")))
                result = w.prepare_job(SimpleNamespace(artifact_root=d), run, source)
                gt.assert_not_called()
                self.assertEqual(cache.call_args.args[0], train_names)
            self.assertEqual(result["datasets"], 6)
            self.assertFalse(result["gt_opened"])
            self.assertTrue(result["ready_for_selector"])
            manifest = json.loads((run/"corpus.json").read_text())
            self.assertEqual(sorted(manifest["datasets"]), sorted(train_names+dev))

    def test_corpus_verifier_requires_current_release_and_complete_partition(self):
        cfg = w.config()
        groups, dev = ["44b6", "6bba"], ["44b6_dev", "6bba_dev"]
        splits = {g: dict(fit=[g+"_fit"], calibration=[g+"_cal"]) for g in groups}
        names = dev+[n for g in splits.values() for ns in g.values() for n in ns]
        h = hashlib.sha256(b"{}").hexdigest()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)/"corpus"
            folder.mkdir()
            dump(Path(d)/cfg["source_run"]/"plan.json", dict(data_root="images"))
            files = {}
            for n in names:
                p = folder/(n+".evidence.npz")
                # Only integrity is read here; open_evidence validates array schemas.
                dump(p, dict(fixture=True))
                files[n] = dict(sha256=digest(p), group=n[:4])
            corpus = dict(schema="selector_corpus_v1", recipe=cfg, code_hash=h, splits=splits,
                development=dev, data_root="images", datasets=files, source_run=cfg["source_run"],
                disposition=w.CORPUS_SCOPE, gt_opened=False, confirmation_evaluated=False)
            dump(folder/"corpus.json", corpus)
            receipt = dict(task="prepare-selector", status="completed", code_hash=h,
                result=dict(ready_for_selector=True, corpus_sha256=digest(folder/"corpus.json")))
            dump(folder/"receipt.json", receipt)
            with patch.object(w, "verify_sources", return_value={}), patch.object(w, "split_corpus", return_value=splits), patch.object(w, "PROTOCOL", dict(development=dev)):
                w.verify_corpus(folder, {})
                bad = dict(corpus, development=dev[:1])
                dump(folder/"corpus.json", bad)
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    w.verify_corpus(folder, {})
                receipt["result"]["corpus_sha256"] = digest(folder/"corpus.json")
                dump(folder/"receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "partition"):
                    w.verify_corpus(folder, {})
                receipt["code_hash"] = "old"
                dump(folder/"receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "current-release"):
                    w.verify_corpus(folder, {})

    def test_selector_sources_cannot_be_promoted_as_experiment_or_package(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dump(root/"runs/selector/receipt.json", dict(status="completed", code_hash="current", task="evaluate-selector"))
            args = SimpleNamespace(source_run="selector", artifact_root=d, task="package", code_hash="current")
            with self.assertRaisesRegex(ValueError, "Expected source task experiment"):
                source_run(args)

    def test_full_workflow_freezes_models_and_all_predictions_before_target_gt(self):
        cfg, events = w.config(), []
        names = ["44b6_dev", "6bba_dev"]
        splits = {g: dict(fit=[g+"_fit"], calibration=[g+"_cal"]) for g in ("44b6", "6bba")}
        corpus = dict(splits=splits, data_root="fake", development=names,
            datasets={n: dict(group=n[:4]) for n in names+['44b6_cal', '6bba_cal']})
        def train(folder, corpus, group, cfg, out):
            events.append(("fit", group))
            return dict(margin=None, training_groups=[group], edge=dict(fit={}), fork=dict(fit={})), {}
        def predict(folder, corpus, name, model, calibration=False):
            if not calibration:
                self.assertTrue((run/"models.json").is_file())
            events.append(("predict", name))
            return graph(), []
        def metric(g, row):
            name = row["name"]
            if name.endswith("_dev"):
                self.assertTrue((run/"development-prediction-manifest.json").is_file())
                self.assertTrue(all(("predict", n) in events for n in names))
            else:
                self.assertTrue((run/"calibration-prediction-manifest.json").is_file())
                self.assertEqual(events[:2], [("fit", "44b6"), ("fit", "6bba")])
            return dict(dataset=name, group=name[:4], num_pred_nodes=len(g.coords))
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)/"run"
            run.mkdir()
            with ExitStack() as stack:
                stack.enter_context(patch.object(w, "verify_corpus", return_value=(corpus, dict(rows=dict(reference=[dict(dataset=n) for n in names])))))
                stack.enter_context(patch.object(w, "train_group", side_effect=train))
                stack.enter_context(patch.object(w, "predict_proposals", side_effect=predict))
                stack.enter_context(patch.object(w, "open_evidence", return_value=(graph(), None, None, None)))
                stack.enter_context(patch.object(w, "metric_row", side_effect=metric))
                stack.enter_context(patch.object(w, "assert_baseline"))
                stack.enter_context(patch.object(w, "tree_fingerprint", return_value="gt"))
                stack.enter_context(patch.object(w, "evaluate_panel", return_value=fake_evaluation(0.)))
                result = w.evaluate_job(SimpleNamespace(), run, Path(d)/"corpus")
            self.assertEqual(result["disposition"], w.SCOPE)
            self.assertFalse(result["uses_gt_for_edits"])
            self.assertFalse(result["confirmation_evaluated"])
            self.assertFalse(result["kaggle_submission_allowed"])
            self.assertEqual(result["next_action"], "rejected_no_kaggle_submission")
            self.assertFalse(any(p.suffix == ".csv" for p in run.iterdir()))


class SelectorRuntimeTests(unittest.TestCase):
    def test_applied_prediction_evaluated_with_real_official_metric(self):
        # Same real runtime dependencies as010; no CUDA, no mocked metric.
        from core.evaluation import make_graph
        from core.production_adapter import as_evaluator_inputs
        from core.mitosis_counterfactual import official_metrics
        g = graph()
        c = ctx(g)
        es, fs = scores(c)
        fs[(0, 2, 3)] = 8.
        tx = a.transaction(0, [2, 3], c, es, fs, w.config())
        repaired, _ = a.apply_transactions(g, [tx], 1., {**w.config(), "max_edit_fraction": 1.})
        def convert(x):
            nodes = {i: dict(t=int(row[0]), z=row[1], y=row[2], x=row[3]) for i, row in enumerate(x.coords)}
            edges = [dict(source_id=int(u), target_id=int(v)) for u, v in x.edges]
            return make_graph(*as_evaluator_inputs(nodes, edges, round_coordinates=False))
        metric = official_metrics()
        counts = metric.evaluate(convert(repaired), convert(repaired), scale=(1., 1., 1.))
        self.assertGreater(counts.edge_tp, 0)
        self.assertEqual(counts.edge_fp, 0)
        self.assertEqual(counts.edge_fn, 0)
