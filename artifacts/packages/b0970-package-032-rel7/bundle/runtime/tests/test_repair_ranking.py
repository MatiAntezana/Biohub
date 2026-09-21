"""012: atomic preferences, zero-anchored ranking, provenance and safe retrieval."""
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

from core.artifacts import digest, dump
from core import repair_ranking as w
from core.nextgen import association as a, repair_ranker as p
from core.nextgen.contracts import Lineage
from test_mitosis_selector import graph, ctx, fake_evaluation


def transaction():
    c = ctx()
    return next(tx for tx in p.transactions([[0, 2, 3]], c) if tx["mother"] == 0), c


def linear(index=-3, value=1.):
    weights = [0.]*len(p.NAMES)
    weights[index] = value
    return dict(feature_names=list(p.NAMES), weights=weights, intercept=0.,
        standardizer=dict(mean=[0.]*len(p.NAMES), scale=[1.]*len(p.NAMES), active=[True]*len(p.NAMES)))


def fixture_corpus():
    splits = {g: dict(fit=[g+"_fit"], calibration=[g+"_cal"]) for g in ('44b6', '6bba')}
    dev = [g+"_dev" for g in splits]
    names = dev + [n for s in splits.values() for ns in s.values() for n in ns]
    return dict(splits=splits, development=dev, data_root="fake-gt",
        datasets={n: dict(group=n[:4]) for n in names})


class RepairPolicyTests(unittest.TestCase):
    def test_transaction_accounts_for_displaced_parent_and_orphan(self):
        tx, c = transaction()
        self.assertEqual(tx['add'], [(0, 3)])
        self.assertEqual(tx['remove'], [(1, 3)])
        self.assertEqual(tx['affected_parents'], [0, 1])
        x = p.features([tx], c)
        np.testing.assert_array_equal(x[0, -3:], [0, 1, 1])

    def test_features_are_exact_whole_graph_factor_difference(self):
        tx, c = transaction()
        x = p.features([tx], c)[0]
        expected_edge = a.edge_features([[0, 3]], c)[0, :-1] - a.edge_features([[1, 3]], c)[0, :-1]
        np.testing.assert_allclose(x[:13], expected_edge, atol=1e-7)
        np.testing.assert_allclose(x[13:31], a.fork_features([[0, 2, 3]], c)[0])

    def test_pruning_removes_fork_potential(self):
        g = graph()
        g.edges = np.asarray([[0, 2], [0, 3], [2, 5], [3, 6], [4, 7]])
        c = ctx(g)
        tx = next(p.transactions(np.empty((0, 3), int), c))
        x = p.features([tx], c)[0]
        np.testing.assert_array_equal(x[-3:], [-1, -1, 0])
        np.testing.assert_allclose(x[13:31], -a.fork_features([[0, 2, 3]], c)[0])

    def test_reference_is_zero_even_with_large_learned_weights(self):
        c = ctx()
        tx = a.transaction(0, [2], c, {}, {}, {'orphan_penalty': 0.})
        x = p.features([tx], c)
        np.testing.assert_array_equal(x, np.zeros((1, len(p.NAMES))))
        self.assertEqual(p.scores(linear(value=100.), x)[0], 0.)

    def test_fork_permutation_does_not_change_transaction_features(self):
        c = ctx()
        first = list(p.transactions([[0, 2, 3]], c))[0]
        second = list(p.transactions([[0, 3, 2]], c))[0]
        np.testing.assert_array_equal(p.features([first], c), p.features([second], c))

    def test_reference_membership_is_not_a_ranker_feature(self):
        tx, c = transaction()
        original = p.features([tx], c)
        c['original'] = set()
        np.testing.assert_array_equal(original, p.features([tx], c))
        self.assertNotIn('reference_edge', ' '.join(p.NAMES))

    def test_temporal_context_is_specific_to_daughters_and_competitor(self):
        tx, c = transaction()
        before = p.features([tx], c)
        c['following'][3] += 20.
        after = p.features([tx], c)
        self.assertFalse(np.array_equal(before, after))

    def test_joint_ranker_can_select_a_reassignment_without_fixed_orphan_cost(self):
        tx, c = transaction()
        rows, audit = p.proposals(np.array([[0, 2, 3]]), c, linear(-2, 2.), w.config())
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['remove'])
        self.assertEqual(rows[0]['gain'], 2.)
        changed, report = a.apply_transactions(c['anchor'], rows, 1., {**w.config(), 'max_edit_fraction': 1.})
        self.assertEqual(report['status'], 'applied')
        self.assertIn((0, 3), map(tuple, changed.edges))
        self.assertNotIn((1, 3), map(tuple, changed.edges))
        self.assertFalse(audit['uses_gt_for_edits'])

    def test_no_positive_gain_or_tie_abstains(self):
        c = ctx()
        rows, _ = p.proposals(np.array([[0, 2, 3], [0, 2, 4]]), c, linear(-2, 2.), w.config())
        self.assertEqual(rows, [])
        rows, _ = p.proposals(np.array([[0, 2, 3]]), c, linear(value=0.), w.config())
        self.assertEqual(rows, [])

    def test_batch_size_does_not_change_ranking(self):
        c = ctx()
        fs = np.array([[0, 2, 3], [1, 3, 4]])
        one = p.proposals(fs, c, linear(-2, 2.), {**w.config(), 'feature_batch_size': 1})
        many = p.proposals(fs, c, linear(-2, 2.), {**w.config(), 'feature_batch_size': 2048})
        self.assertEqual(one, many)
        with self.assertRaises(ValueError):
            list(p.batches([], 0))

    def test_zero_anchor_and_finite_validation(self):
        m = linear()
        for bad in ({**m, 'intercept': 1.}, {**m, 'standardizer': {**m['standardizer'], 'mean': [1.]*len(p.NAMES)}}):
            with self.assertRaises(ValueError):
                p.scores(bad, np.zeros((1, len(p.NAMES))))
        with self.assertRaises(ValueError):
            p.scores(m, np.full((1, len(p.NAMES)), np.nan))

    def test_inference_has_no_annotation_or_image_io_interface(self):
        source = inspect.getsource(p)
        for forbidden in ('match_annotation', '.geff', 'ImageSource', 'torch', 'read_text', 'np.load'):
            self.assertNotIn(forbidden, source)
        self.assertEqual(list(inspect.signature(p.proposals).parameters), ['forks', 'ctx', 'model', 'cfg'])


class RepairWorkflowTests(unittest.TestCase):
    def test_preferences_require_every_changed_edge_known(self):
        tx, _ = transaction()
        mapping = {10: 0, 11: 1, 12: 2, 13: 3}
        self.assertEqual(w.preference(tx, mapping, {12: 10, 13: 10}), 1)
        self.assertEqual(w.preference(tx, mapping, {12: 10, 13: 11}), -1)
        self.assertIsNone(w.preference(tx, {10: 0, 12: 2, 13: 3}, {12: 10, 13: 10}))
        self.assertIsNone(w.preference(tx, mapping, {12: 10}))

    def test_mixed_tradeoff_not_a_fabricated_negative(self):
        tx = dict(add=[(0, 2), (0, 3)], remove=[])
        self.assertEqual(w.preference(tx, {10: 0, 12: 2, 13: 3}, {12: 10, 13: 11}), 0)
        self.assertEqual(w.preference(dict(add=[], remove=[]), {}, {}), 0)

    def test_mother_weighting_prevents_dense_duplicate_domination(self):
        np.testing.assert_allclose(w.mother_weights(['a', 'a', 'b']), [.25, .25, .5])
        with self.assertRaises(ValueError):
            w.mother_weights([])

    def test_rank_objective_gradient_and_extreme_logits(self):
        rng = np.random.default_rng(3)
        x, weights, theta = rng.normal(size=(7, 4)), np.full(7, 1/7), rng.normal(size=4)
        _, gradient = w.rank_objective(theta, x, weights, 1.)
        h = 1e-5
        numeric = [(w.rank_objective(theta+np.eye(4)[i]*h, x, weights, 1.)[0]-
            w.rank_objective(theta-np.eye(4)[i]*h, x, weights, 1.)[0])/(2*h) for i in range(4)]
        np.testing.assert_allclose(gradient, numeric, atol=1e-7)
        loss, grad = w.rank_objective(theta*10000, x, weights, 1.)
        self.assertTrue(np.isfinite(loss) and np.isfinite(grad).all())

    def test_two_fixed_fit_rounds_preserve_reversal_and_zero(self):
        x = np.zeros((12, len(p.NAMES)))
        x[:, 0] = np.arange(1, 13)
        y = np.array([-1.]*6+[1.]*6)
        x[:6, 0] *= -1
        keys = ['m'+str(i//2) for i in range(12)]
        m = w.fit_ranker(x, y, keys, w.config())
        self.assertEqual(len(m['fit_rounds']), 2)
        self.assertEqual(len(m['fit_rounds'][1]['hard_indices']), 6)
        self.assertEqual(m['intercept'], 0.)
        self.assertFalse(m['standardizer']['active'][1])
        np.testing.assert_allclose(p.scores(m, -x), -p.scores(m, x))
        self.assertTrue((y*p.scores(m, x) > 0).all())
        self.assertEqual(p.scores(m, np.zeros((1, len(p.NAMES))))[0], 0.)

    def test_unknown_or_nonfinite_preferences_are_refused(self):
        x = np.zeros((12, len(p.NAMES)))
        for y in (np.zeros(12), np.full(12, np.nan)):
            with self.assertRaises(ValueError):
                w.fit_ranker(x, y, list(range(12)), w.config())
        with self.assertRaises(ValueError):
            w.fit_ranker(x+np.nan, np.ones(12), list(range(12)), w.config())

    def test_supervision_guard_does_not_count_inverse_preferences_as_new_divisions(self):
        cfg = w.config()
        self.assertFalse(w.sufficient(dict(preferences=1000, mothers=30, confirmed_true_forks=2), cfg))
        self.assertFalse(w.sufficient(dict(preferences=1000, mothers=1, confirmed_true_forks=10), cfg))
        self.assertTrue(w.sufficient(dict(preferences=10, mothers=10, confirmed_true_forks=3), cfg))

    def test_single_calibration_policy_can_only_be_enabled_or_disabled(self):
        cfg = w.config()
        self.assertEqual(cfg['margin_grid'], [1.])
        trial = dict(evaluation=fake_evaluation(.005), budget_rejected=False)
        self.assertEqual(w.previous.select_margin({1.: trial}, cfg), 1.)
        trial['budget_rejected'] = True
        self.assertIsNone(w.previous.select_margin({1.: trial}, cfg))
        with self.assertRaises(ValueError):
            w.previous.select_margin({1.: trial, .5: trial}, cfg)

    def test_model_excludes_target_from_fit_and_calibration(self):
        corpus = fixture_corpus()
        model = dict(schema='repair_ranker_012', scope=w.SCOPE, recipe=w.config(),
            training_groups=['44b6'], training_datasets=['44b6_fit'], selection_groups=[], selection_datasets=[])
        w.check_model(model, corpus, '44b6_cal', calibration=True)
        with self.assertRaises(ValueError):
            w.check_model(model, corpus, '6bba_cal', calibration=True)
        model.update(selection_groups=['44b6'], selection_datasets=['44b6_cal'])
        w.check_model(model, corpus, '6bba_dev')
        with self.assertRaises(ValueError):
            w.check_model(model, corpus, '44b6_dev')
        with self.assertRaises(ValueError):
            w.check_model({**model, 'selection_groups': ['6bba']}, corpus, '6bba_dev')

    def test_training_collection_uses_fit_only_and_retains_whole_reassignment(self):
        corpus = fixture_corpus()
        mapping = {10: 0, 11: 1, 12: 2, 13: 3}
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            opened = stack.enter_context(patch.object(w.previous, 'open_evidence',
                return_value=(graph(), None, np.array([[0, 2, 3], [0, 2, 4]]), ctx())))
            stack.enter_context(patch.object(w.previous, 'match_annotation', return_value=(mapping, None, {12: 10, 13: 10}, None)))
            stack.enter_context(patch.object(w.previous, 'tree_fingerprint', return_value='unchanged'))
            x, y, keys, evidence, stats = w.collect_group(Path(d), corpus, '44b6', Path(d))
            self.assertEqual(opened.call_args.args[1], '44b6_fit')
            self.assertEqual(y.tolist(), [1.])
            self.assertEqual(keys, ['44b6_fit:0'])
            self.assertEqual(stats['beneficial'], 1)
            self.assertEqual(evidence['44b6_fit']['fork_training_screened_out'], 1)
            np.testing.assert_array_equal(x[0, -3:], [0, 1, 1])

    def test_inference_passes_full_proposal_universe_not_training_screen(self):
        corpus = fixture_corpus()
        model = dict(schema='repair_ranker_012', scope=w.SCOPE, recipe=w.config(), ranker=linear(),
            training_groups=['44b6'], training_datasets=['44b6_fit'],
            selection_groups=['44b6'], selection_datasets=['44b6_cal'])
        forks = np.array([[0, 2, 3], [0, 2, 4], [1, 3, 4]])
        with patch.object(w.previous, 'open_evidence', return_value=(graph(), None, forks, ctx())), \
                patch.object(w.policy, 'proposals', return_value=([], {})) as proposals:
            w.predict(Path('corpus'), corpus, '6bba_dev', model)
            self.assertIs(proposals.call_args.args[0], forks)

    def test_parent_manifest_accepts_only_named_additions_and_driver_changes(self):
        old = {'core/production.py': 'stable', 'ops/biohub.py': 'old'}
        cfg = {**w.config(), 'source_release': hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()}
        current = {**old, 'ops/biohub.py': 'new', **dict.fromkeys(w.ADDITIONS, 'new')}
        with patch.object(w, 'config', return_value=cfg):
            self.assertEqual(w.parent_manifest(current, old), old)
            for bad in ({**current, 'core/production.py': 'changed'}, {**current, 'other.py': 'extra'}):
                with self.assertRaises(ValueError):
                    w.parent_manifest(bad, old)
            with self.assertRaises(ValueError):
                w.parent_manifest(current, {**old, 'ops/biohub.py': 'tampered'})

    def test_insufficient_supervision_never_opens_calibration_or_predicts(self):
        corpus = fixture_corpus()
        data = (np.empty((0, len(p.NAMES))), np.array([]), [], {},
            dict(preferences=0, mothers=0, confirmed_true_forks=0))
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            stack.enter_context(patch.object(w, 'verify_sources', return_value=(corpus, {})))
            stack.enter_context(patch.object(w, 'collect_group', return_value=data))
            predict = stack.enter_context(patch.object(w, 'predict'))
            fit = stack.enter_context(patch.object(w, 'fit_ranker'))
            result = w.evaluate_job(SimpleNamespace(), Path(d), Path(d)/'source')
            self.assertEqual(result['next_action'], 'insufficient_supervision_no_submission')
            self.assertFalse(result['model_trained'])
            self.assertFalse(result['tracking_evaluated'])
            predict.assert_not_called()
            fit.assert_not_called()

    def test_full_workflow_orders_fit_predictions_and_gt_and_blocks_publication(self):
        corpus, events = fixture_corpus(), []
        def collect(folder, corpus, group, run):
            events.append(('collect', group))
            return np.ones((12, len(p.NAMES))), np.ones(12), list(range(12)), {}, dict(preferences=12, mothers=12, confirmed_true_forks=3)
        def fit(*args):
            events.append(('fit', None))
            return linear()
        def predict(folder, corpus, name, model, calibration=False):
            if calibration:
                self.assertEqual(sum(e[0] == 'fit' for e in events), 2)
            else:
                self.assertTrue((run/'models.json').is_file())
            events.append(('predict', name))
            return graph(), [], dict(uses_gt_for_edits=False)
        def metric(g, row):
            name = row['name']
            if name.endswith('_cal'):
                self.assertTrue((run/'calibration-prediction-manifest.json').is_file())
            else:
                self.assertTrue((run/'development-prediction-manifest.json').is_file())
                self.assertTrue(all(('predict', n) in events for n in corpus['development']))
            return dict(dataset=name, group=name[:4], num_pred_nodes=len(g.coords))
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run = Path(d)
            stack.enter_context(patch.object(w, 'verify_sources', return_value=(corpus, dict(rows=dict(reference=[dict(dataset=n) for n in corpus['development']])))))
            stack.enter_context(patch.object(w, 'collect_group', side_effect=collect))
            stack.enter_context(patch.object(w, 'fit_ranker', side_effect=fit))
            stack.enter_context(patch.object(w, 'predict', side_effect=predict))
            stack.enter_context(patch.object(w.previous, 'open_evidence', return_value=(graph(), None, None, None)))
            stack.enter_context(patch.object(w.previous, 'resolve_dataset_row', side_effect=lambda n, *a, **k: dict(name=n, gt=n+'.geff')))
            stack.enter_context(patch.object(w.previous, 'metric_row', side_effect=metric))
            stack.enter_context(patch.object(w.previous, 'assert_baseline'))
            stack.enter_context(patch.object(w.previous, 'tree_fingerprint', return_value='gt'))
            stack.enter_context(patch.object(w.previous, 'evaluate_panel', return_value=fake_evaluation(0.)))
            result = w.evaluate_job(SimpleNamespace(), run, run/'source')
            self.assertEqual(result['next_action'], 'rejected_no_kaggle_submission')
            for key in ('confirmation_evaluated', 'visible_evaluated', 'kaggle_submission_allowed', 'uses_gt_for_edits'):
                self.assertFalse(result[key])
            self.assertTrue(result['calibration_reused_from011'])
            self.assertFalse(list(run.glob('*.csv')))

    def test_cpu_task_does_not_relax_gpu_jobs_and_wrong_node_is_refused(self):
        from ops.biohub import child_environment, resource_profile, launch_job
        self.assertEqual(resource_profile('rank-repairs'), 'cpu_evaluation_v1')
        self.assertEqual(child_environment('rank-repairs')['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(resource_profile('prepare-selector'), 'gpu_exclusive_v1')
        with self.assertRaisesRegex(ValueError, 'A1555'):
            launch_job(SimpleNamespace(task='rank-repairs', node='A1688'), Path('unused'), [])

    def test_remote_requires_current_complete_tests_and_cannot_package_ranker(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'verify_sources'):
            root = Path(d)/'runs'
            dump(root/'corpus/receipt.json', dict(status='completed', task='prepare-selector', code_hash='old'))
            args = SimpleNamespace(task='rank-repairs', artifact_root=d, source_run='corpus',
                tests_run='tests', code_hash='current', node='A1555')
            result = dict(tests_passed=True, tests_run=168, tests_expected=168, suite='full_discovery',
                skipped=0, errors=0, failures=0)
            receipt = dict(task='tests', status='completed', code_hash='current', node='A1555', result=result)
            dump(root/'tests/receipt.json', receipt)
            self.assertEqual(source_run(args), root/'corpus')
            for bad in ({**result, 'skipped': 1}, {**result, 'tests_expected': 200}, {**result, 'suite': 'local_cpu'}):
                dump(root/'tests/receipt.json', {**receipt, 'result': bad})
                with self.assertRaises(ValueError):
                    source_run(args)
            dump(root/'ranker/receipt.json', dict(task='rank-repairs', status='completed', code_hash='current'))
            args.source_run, args.task = 'ranker', 'package'
            with self.assertRaisesRegex(ValueError, 'Expected source task experiment'):
                source_run(args)


class RetrievalTests(unittest.TestCase):
    def test_resume_is_explicit_and_preserves_existing_files(self):
        from ops.biohub import fetch_destination, fetch_command, DEFAULT_NODES
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d).resolve()/'run'
            dest.mkdir()
            dump(dest/'receipt.json', dict(status='completed', node='A1555', run_id='run'))
            with self.assertRaises(FileExistsError):
                fetch_destination(dest, 'A1555', 'run', False)
            snapshot = fetch_destination(dest, 'A1555', 'run', True)
            self.assertEqual(snapshot[dest/'receipt.json'], digest(dest/'receipt.json'))
            cmd = fetch_command(DEFAULT_NODES['A1555'], 'run', dest, True)
            self.assertIn('--ignore-existing', cmd)
            self.assertNotIn('--delete', cmd)
            self.assertIn('--no-links', cmd)
            self.assertFalse(any('ProxyCommand' in arg for arg in cmd))
            self.assertIn('StrictHostKeyChecking=yes', ' '.join(cmd))

    def test_resume_refuses_wrong_run_running_receipt_and_symlinks(self):
        from ops.biohub import fetch_destination
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d).resolve()/'run'
            dest.mkdir()
            for receipt in (dict(status='running', node='A1555', run_id='run'),
                            dict(status='completed', node='A1688', run_id='run'),
                            dict(status='completed', node='A1555', run_id='other')):
                dump(dest/'receipt.json', receipt)
                with self.assertRaises(ValueError):
                    fetch_destination(dest, 'A1555', 'run', True)
            (dest/'link').symlink_to(Path(d)/'missing')
            with self.assertRaises(ValueError):
                fetch_destination(dest, 'A1555', 'run', True)

    def test_source_corpus_wrong_id_is_refused_before_reading_any_data(self):
        with patch.object(w, 'audit_inputs') as audit:
            with self.assertRaises(ValueError):
                w.verify_sources(Path('other'), {})
            audit.assert_not_called()

    def test_partial_retrieval_fails_closed(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'parent_manifest', return_value={}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                w.audit_inputs(Path(d))

    def test_all_61_input_arrays_are_verified_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            folder, result_dir = root/'corpus', root/'b0970-selector-011'
            folder.mkdir()
            result_dir.mkdir()
            development = [f'44b6_dev{i}' for i in range(8)]
            names = development + [f'6bba_fit{i}' for i in range(45)]
            datasets, predictions = {}, {}
            for name in names:
                path = folder/(name+'.evidence.npz')
                with path.open('xb') as handle:
                    np.savez(handle, coords=graph().coords, reference_edges=graph().edges)
                datasets[name] = dict(sha256=digest(path))
            for name in development:
                path = result_dir/(name+'.prediction.npz')
                with path.open('xb') as handle:
                    np.savez(handle, coords=graph().coords, edges=graph().edges)
                predictions[name] = dict(sha256=digest(path))
            dump(folder/'receipt.json', dict(code_manifest={}))
            dump(folder/'corpus.json', dict(datasets=datasets, development=development))
            dump(result_dir/'selector.json', dict(calibration_margins={'44b6': None, '6bba': None},
                confirmation_evaluated=False, kaggle_submission_allowed=False))
            dump(result_dir/'models.json', {})
            dump(result_dir/'calibration-prediction-manifest.json', {})
            dump(result_dir/'development-prediction-manifest.json', predictions)
            artifacts = {str(path.relative_to(root)): digest(path) for path in root.rglob('*.json')}
            cfg = {**w.config(), 'source_run': 'corpus', 'source_release': hashlib.sha256(b'{}').hexdigest(),
                'source_artifacts': artifacts}
            stack.enter_context(patch.object(w, 'config', return_value=cfg))
            stack.enter_context(patch.object(w, 'code_manifest', return_value={}))
            forbidden = stack.enter_context(patch.object(w.previous, 'match_annotation', side_effect=AssertionError('No GT during retrieval')))
            report, _ = w.audit_inputs(root)
            self.assertTrue(report['passed'])
            self.assertEqual(report['evidence_npz'], 53)
            self.assertEqual(report['prediction_npz'], 8)
            forbidden.assert_not_called()
            (folder/(names[-1]+'.evidence.npz')).unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                w.audit_inputs(root)


class RepairRuntimeTests(unittest.TestCase):
    def test_ranker_reassignment_with_real_official_runtime(self):
        from core.evaluation import make_graph
        from core.production_adapter import as_evaluator_inputs
        from core.mitosis_counterfactual import official_metrics
        c = ctx()
        rows, _ = p.proposals(np.array([[0, 2, 3]]), c, linear(-2, 2.), w.config())
        g, audit = a.apply_transactions(c['anchor'], rows, 1., {**w.config(), 'max_edit_fraction': 1.})
        self.assertEqual(audit['status'], 'applied')
        def convert():
            nodes = {i: dict(t=int(v[0]), z=v[1], y=v[2], x=v[3]) for i, v in enumerate(g.coords)}
            edges = [dict(source_id=int(u), target_id=int(v)) for u, v in g.edges]
            return make_graph(*as_evaluator_inputs(nodes, edges, round_coordinates=False))
        counts = official_metrics().evaluate(convert(), convert(), scale=(1., 1., 1.))
        self.assertGreater(counts.edge_tp, 0)
        self.assertEqual(counts.edge_fp, 0)
        self.assertEqual(counts.edge_fn, 0)
