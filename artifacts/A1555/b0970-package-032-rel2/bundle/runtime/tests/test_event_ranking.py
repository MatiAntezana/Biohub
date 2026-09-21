"""013 toy-fixture tests only; no real local fitting or detector inference."""
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

from core import event_ranking as w
from core.artifacts import dump, digest
from core.nextgen import event_ranker as p, repair_ranker as r
from test_mitosis_selector import graph, ctx
from test_repair_ranking import linear, fixture_corpus


def toy_data():
    x = np.zeros((3, len(r.NAMES)))
    x[:, 0] = [1, -1, 1]
    return dict(x=x, y=np.array([1., -1., 1.]),
        keys=np.array(['v:1', 'v:2', 'v:1']), strata=np.array(['beneficial', 'harmful', 'event']))


def annotation():
    mapping = {100+i: i for i in range(8)}
    children = {100: {102, 103}, 101: {104}, 102: {105}, 103: {106}, 104: {107}}
    parents = {c: m for m, ds in children.items() for c in ds}
    return mapping, children, parents


def table(scores=None):
    values = np.array([[.8, -.1], [.2, -.3], [1.2, .2], [.1, .1]]) if scores is None else np.asarray(scores)
    return dict(ids=np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7], [3, 8, 9]]),
        scores=values, changed=np.ones(4, bool), labels=np.array([1, -1, 2, 0]), fork_ranks=[])


class EventPolicyTests(unittest.TestCase):
    def test_near_tie_is_order_independent(self):
        ids = np.array([[0, 2, 3], [0, 2, 4], [0, 3, 4]])
        scores = np.array([1., 1.+.5e-10, 1.+1.25e-10])
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            self.assertEqual(p.best_positive(ids[order], scores[order], np.ones(3, bool)).tolist(), [])

    def test_invalid_batch_or_empty_models_rejected(self):
        for models, batch in (([linear()], 0), ([], 1), ([linear()], -1)):
            with self.assertRaises(ValueError):
                p.score_universe([[0, 2, 3]], ctx(), models, batch)

    def test_all_forks_and_existing_prunes_retained(self):
        c = ctx()
        forks = np.array([[0, 2, 3], [0, 2, 4]])
        np.testing.assert_array_equal(p.hypothesis_ids(forks, c), forks)
        c['children'][0] = {2, 3}
        self.assertEqual(p.hypothesis_ids(forks, c)[-2:].tolist(), [[0, 2, -1], [0, 3, -1]])

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            p.hypothesis_ids([[0, 2, 3], [0, 2, 3]], ctx())

    def test_scores_keep_negatives_and_pair_the_comparator(self):
        ids, scores, changed = p.score_universe([[0, 2, 3]], ctx(), [linear(-2, 1), linear(-2, -1)], 1)
        self.assertEqual(scores.tolist(), [[1., -1.]])
        self.assertEqual(ids.tolist(), [[0, 2, 3]])
        self.assertTrue(changed.all())

    def test_intact_fork_has_zero_and_is_not_a_proposed_edit(self):
        g = graph()
        g.edges = np.array([[0, 2], [0, 3]])
        ids, scores, changed = p.score_universe([[0, 2, 3]], ctx(g), [linear(-2, 10)], 2)
        self.assertEqual(scores[0, 0], 0.)
        self.assertFalse(changed[0])
        self.assertEqual(p.best_positive(ids, scores[:, 0], changed).tolist(), [])

    def test_best_positive_rejects_ties_and_noops(self):
        ids = np.array([[0, 2, 3], [0, 2, 4], [1, 3, 4]])
        self.assertEqual(p.best_positive(ids, [1, 1, 2], np.array([True, True, False])).tolist(), [])
        self.assertEqual(p.best_positive(ids, [2, 1, -2], np.ones(3, bool)).tolist(), [0])

    def test_full_score_batching_is_identical(self):
        args = ([[0, 2, 3], [0, 2, 4], [1, 3, 4]], ctx(), [linear(-2, 1)])
        for a, b in zip(p.score_universe(*args, 1), p.score_universe(*args, 2048)):
            np.testing.assert_array_equal(a, b)

    def test_reference_and_predictor_never_modify_source(self):
        c = ctx()
        original = c['anchor'].edges.copy()
        tx = p.reference(0)
        np.testing.assert_array_equal(r.features([tx], c), np.zeros((1, len(r.NAMES))))
        p.score_universe([[0, 2, 3]], c, [linear()], 1)
        np.testing.assert_array_equal(c['anchor'].edges, original)

    def test_policy_has_no_gt_training_or_editing_interface(self):
        code = inspect.getsource(p)
        for forbidden in ('match_annotation', '.geff', 'np.load', 'read_text', 'apply_transactions', 'minimize', 'torch'):
            self.assertNotIn(forbidden, code)


class EventLearningTests(unittest.TestCase):
    def test_keep_is_not_duplicated_if_reference_fork_is_already_an_alternative(self):
        g = graph()
        g.edges = np.array([[0, 2], [0, 4], [1, 3]])
        mapping, children, parents = annotation()
        _, rows, _ = w.event_examples('toy', [[0, 2, 3], [0, 2, 4]], ctx(g), mapping, children, parents)
        self.assertTrue(any(row['alternative'] == [0, 2, 4] for row in rows))
        self.assertFalse(any(row['alternative'] == [0, -1, -1] for row in rows))

    def test_direct_class_masses_are_fixed_despite_imbalance(self):
        weights = w.event_weights(['b']+['h'+str(i) for i in range(1000)]+['e'],
            ['beneficial']+['harmful']*1000+['event'], w.config())
        self.assertAlmostEqual(weights[0], .25)
        self.assertAlmostEqual(weights[1:-1].sum(), .25)
        self.assertAlmostEqual(weights[-1], .5)

    def test_duplicate_alternatives_do_not_add_event_weight(self):
        weights = w.event_weights(['b', 'h', 'e1', 'e1', 'e2'],
            ['beneficial', 'harmful', 'event', 'event', 'event'], w.config())
        np.testing.assert_allclose(weights, [.25, .25, .125, .125, .25])

    def test_absent_stratum_does_not_silently_reweight_remaining_data(self):
        with self.assertRaises(ValueError):
            w.event_weights(['b', 'h'], ['beneficial', 'harmful'], w.config())

    def test_gradient_and_one_convex_fit_on_toy_only(self):
        data = toy_data()
        model = w.fit_event_ranker(data, w.config())
        self.assertEqual(model['fit']['fits'], 1)
        self.assertEqual(model['fit']['stratum_mass'], w.config()['stratum_mass'])
        self.assertTrue((data['y']*r.scores(model, data['x']) > 0).all())
        self.assertFalse(model['standardizer']['active'][1])
        np.testing.assert_allclose(r.scores(model, -data['x']), -r.scores(model, data['x']))

    def test_bad_signs_and_nonfinite_training_are_rejected(self):
        for bad in ({**toy_data(), 'y': np.zeros(3)}, {**toy_data(), 'x': toy_data()['x']+np.nan}):
            with self.assertRaises(ValueError):
                w.fit_event_ranker(bad, w.config())

    def test_failed_optimizer_does_not_produce_checkpoint(self):
        bad = SimpleNamespace(x=np.zeros(len(r.NAMES)), success=False, message='failed', nit=0)
        with patch.object(w, 'minimize', return_value=bad), self.assertRaises(ValueError):
            w.fit_event_ranker(toy_data(), w.config())

    def test_report_reveals_all_repair_positives_rejected(self):
        data = toy_data()
        metrics = w.preference_metrics(data, linear(0, -1))
        self.assertEqual(metrics['beneficial']['satisfied'], 0)
        self.assertEqual(metrics['beneficial']['fraction'], 0.)

    def test_guard_counts_events_and_volumes_not_contrasts(self):
        summary = w.training_summary(toy_data())
        self.assertEqual(summary['unique_mothers_across_strata'], 2)
        self.assertFalse(w.sufficient(summary, w.config()))
        summary = {k: dict(events=100, volumes=1) for k in ('beneficial', 'harmful', 'event')}
        self.assertFalse(w.sufficient(summary, w.config()))

    def test_net_edge_comparison_handles_reassignment(self):
        mapping, _, parents = annotation()
        tx = p.transaction([0, 2, 3], ctx())
        truth = w.edge_truth(mapping, parents)
        self.assertEqual(w.contrast_label(tx, p.reference(0), truth), 1)
        self.assertEqual(w.contrast_label(p.reference(0), tx, truth), -1)

    def test_shared_unknown_changes_cancel_but_different_unknowns_do_not(self):
        good = dict(add=[(0, 2), (9, 10)], remove=[])
        bad = dict(add=[(9, 10)], remove=[])
        truth = w.edge_truth({100: 0, 102: 2}, {102: 100})
        self.assertEqual(w.contrast_label(good, bad, truth), 1)
        self.assertEqual(w.contrast_label(good, p.reference(0), truth), w.UNKNOWN)

    def test_mixed_and_noop_are_not_harmful_labels(self):
        truth = w.edge_truth({100: 0, 102: 2, 103: 3}, {102: 100, 103: 999})
        self.assertEqual(w.contrast_label(dict(add=[(0, 2), (0, 3)], remove=[]), p.reference(0), truth), w.MIXED)
        self.assertEqual(w.contrast_label(p.reference(0), p.reference(0), truth), w.NOOP)

    def test_reference_correct_division_provides_training_comparisons(self):
        g = graph()
        g.edges = np.array([[0, 2], [0, 3], [2, 5], [3, 6], [4, 7]])
        mapping, children, parents = annotation()
        x, rows, stats = w.event_examples('44b6_toy', [[0, 2, 3]], ctx(g), mapping, children, parents)
        self.assertEqual(stats['covered_gt_forks'], 1)
        self.assertEqual(stats['usable_events'], 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['reference_correct'] for row in rows))
        self.assertTrue((x[:, -2] == 1).all())

    def test_new_fork_compares_against_keep_without_inventing_nodes(self):
        c = ctx()
        before = c['anchor'].edges.copy()
        mapping, children, parents = annotation()
        _, rows, stats = w.event_examples('44b6_toy', [[0, 2, 3], [0, 2, 4]], c, mapping, children, parents)
        self.assertTrue(any(row['alternative'] == [0, -1, -1] for row in rows))
        self.assertEqual(stats['usable_events'], 1)
        np.testing.assert_array_equal(c['anchor'].edges, before)

    def test_uncovered_gt_never_expands_the_frozen_pool(self):
        mapping, children, parents = annotation()
        x, rows, stats = w.event_examples('toy', [[0, 2, 4]], ctx(), mapping, children, parents)
        self.assertEqual(len(x), 0)
        self.assertEqual(rows, [])
        self.assertEqual(stats['outside_frozen_pool'], 1)


class EventDiagnosticTests(unittest.TestCase):
    def test_reference_competitor_is_not_counted_twice_in_rank(self):
        mapping, children, _ = annotation()
        rows = w.fork_ranks(np.array([[0, 2, 3], [0, 2, 4]]),
            np.array([[-1, -1], [0, 0]]), np.array([True, False]), mapping, children)
        self.assertEqual(rows[0]['models']['event013']['rank_worst'], 2)

    def test_unknowns_are_not_confirmed_false_positives(self):
        result = w.summarize_tables({'toy': table()}, 0, .5)
        self.assertEqual(result['unique_best_selected']['beneficial'], 1)
        self.assertEqual(result['unique_best_selected']['unknown'], 1)
        self.assertEqual(result['confirmed_only']['fp'], 0)
        self.assertEqual(result['confirmed_only']['auroc'], 1.)
        self.assertEqual(result['unknown_fraction_selected'], .5)

    def test_auc_ties_are_half_credit(self):
        t = table()
        t['scores'][:2, 0] = .2
        self.assertEqual(w.summarize_tables({'toy': t}, 0, 0.)['confirmed_only']['auroc'], .5)

    def test_empty_calibration_or_one_class_abstains(self):
        result = w.calibrate({}, w.config())
        self.assertIsNone(result['diagnostic_margin'])
        self.assertFalse(result['eligible'])

    def test_margin_follows_source_score_scale_not_old_constant(self):
        cfg = {**w.config(), 'calibration_minimum_harmful': 1}
        result = w.calibrate({'toy': table()}, cfg)
        self.assertAlmostEqual(result['diagnostic_margin'], .2)
        self.assertFalse(result['authorizes_graph_edits'])
        self.assertFalse(result['precision_guarantee'])

    def test_calibration_score_rescaling_rescales_margin(self):
        cfg = {**w.config(), 'calibration_minimum_harmful': 1}
        t = table()
        t['scores'][:, 0] *= 10
        self.assertAlmostEqual(w.calibrate({'toy': t}, cfg)['diagnostic_margin'], 2.)

    def test_mixed_selected_proposal_vetoes_margin(self):
        t = table()
        t['scores'][3, 0] = .9
        result = w.calibrate({'toy': t}, {**w.config(), 'calibration_minimum_harmful': 1})
        self.assertIsNone(result['diagnostic_margin'])
        self.assertTrue(any('mixed' in r for r in result['reasons']))

    def test_fork_rank_competes_with_unlabeled_alternatives_and_keep(self):
        mapping, children, _ = annotation()
        ids = np.array([[0, 2, 3], [0, 2, 4]])
        values = np.array([[.5, -.1], [.8, -.2]])
        rows = w.fork_ranks(ids, values, np.ones(2, bool), mapping, children)
        self.assertEqual(rows[0]['models']['event013']['rank_worst'], 2)
        self.assertEqual(rows[0]['models']['ranker012']['rank_worst'], 2)

    def test_intact_fork_does_not_tie_with_duplicate_keep(self):
        mapping, children, _ = annotation()
        rows = w.fork_ranks(np.array([[0, 2, 3], [0, 2, -1]]),
            np.array([[0, 0], [-1, -1]]), np.array([False, True]), mapping, children)
        self.assertTrue(rows[0]['already_present'])
        self.assertTrue(rows[0]['models']['event013']['unique_top1'])

    def test_gate_rejects_no_improvement_even_with_high_auc(self):
        m = w.summarize_tables({'toy': table()}, 0, .5)
        row = dict(event013_at_zero=m, ranker012_at_zero=m, event013_at_source_margin=m)
        self.assertFalse(w.discrimination_gate({'44b6': row})['passed'])

    def test_fitted_and_calibration_prefix_exclusion(self):
        corpus = fixture_corpus()
        model = dict(schema='event_ranker_013', scope=w.SCOPE, recipe=w.config(),
            training_groups=['44b6'], training_datasets=['44b6_fit'],
            selection_groups=[], selection_datasets=[], diagnostic_margin=None)
        w.check_model(model, corpus, '44b6_cal', True)
        with self.assertRaises(ValueError):
            w.check_model(model, corpus, '6bba_cal', True)
        model.update(selection_groups=['44b6'], selection_datasets=['44b6_cal'])
        w.check_model(model, corpus, '6bba_dev')
        with self.assertRaises(ValueError):
            w.check_model(model, corpus, '44b6_dev')


class EventWorkflowTests(unittest.TestCase):
    def test_training_collection_reuses_direct_preferences_and_only_fit_annotations(self):
        corpus = fixture_corpus()
        mapping, children, parents = annotation()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            folder = Path(d)/'source'
            folder.mkdir()
            direct = toy_data()
            direct['keys'] = np.array(['44b6_fit:0', '44b6_fit:1', '44b6_fit:0'])
            with (folder/'preferences-44b6.npz').open('xb') as handle:
                np.savez(handle, differences=direct['x'][:2], signs=direct['y'][:2], mother_keys=direct['keys'][:2])
            dump(folder/'preferences-44b6.json', dict(datasets={'44b6_fit': dict(gt_sha256='gt')}))
            opened = stack.enter_context(patch.object(w.source, 'open_evidence',
                return_value=(graph(), None, np.array([[0, 2, 3]]), ctx())))
            stack.enter_context(patch.object(w.source, 'tree_fingerprint', return_value='gt'))
            matched = stack.enter_context(patch.object(w.source, 'match_annotation',
                return_value=(mapping, children, parents, None)))
            data, summary, hashes = w.collect_training(folder, corpus, '44b6', Path(d))
            self.assertEqual(opened.call_args.args[1], '44b6_fit')
            self.assertEqual(matched.call_args.args[1], Path('fake-gt/44b6_fit.geff'))
            self.assertEqual(hashes, {'44b6_fit': 'gt'})
            self.assertEqual(summary['event']['events'], 1)
            np.testing.assert_array_equal(data['x'][:2], direct['x'][:2])
            self.assertTrue((Path(d)/'training-44b6.npz').is_file())

    def test_prediction_evaluation_checks_full_universe_and_changed_flags(self):
        corpus = fixture_corpus()
        mapping, children, parents = annotation()
        forks = np.array([[0, 2, 3]])
        table = dict(ids=forks, scores=np.array([[.5, -.5]]), changed=np.array([True]))
        with patch.object(w.source, 'open_evidence', return_value=(graph(), None, forks, ctx())), \
                patch.object(w.source, 'tree_fingerprint', return_value='gt'), \
                patch.object(w.source, 'match_annotation', return_value=(mapping, children, parents, None)):
            data, sha = w.evaluate_prediction(Path('source'), corpus, '44b6_dev', table)
            self.assertEqual(data['labels'].tolist(), [1])
            self.assertEqual(sha, 'gt')
            with self.assertRaises(AssertionError):
                w.evaluate_prediction(Path('source'), corpus, '44b6_dev', {**table, 'changed': np.array([False])})

    def test_archive_identity_and_unchanged_science_required(self):
        old = {'core/production.py': 'stable', 'ops/biohub.py': 'old'}
        cfg = {**w.config(), 'source_release': hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()}
        current = {**old, **dict.fromkeys(w.ADDITIONS, 'new'), 'ops/biohub.py': 'new'}
        with patch.object(w, 'config', return_value=cfg):
            self.assertEqual(w.parent_manifest(current, old), old)
            for bad in ({**current, 'core/production.py': 'changed'}, {**current, 'extra.py': 'unapproved'}):
                with self.assertRaises(ValueError):
                    w.parent_manifest(bad, old)

    def test_wrong_source_id_rejected_before_io(self):
        with patch.object(w, 'audit_inputs') as audit, self.assertRaises(ValueError):
            w.verify_sources(Path('other'), {})
        audit.assert_not_called()

    def test_partial_input_retrieval_fails_closed(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            w.audit_inputs(Path(d))

    def test_insufficient_events_never_fit_or_open_calibration(self):
        corpus = fixture_corpus()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run, folder = Path(d), Path(d)/'source'
            folder.mkdir()
            for group in corpus['splits']:
                dump(folder/('ranker-'+group+'.json'), {})
            stack.enter_context(patch.object(w, 'verify_sources', return_value=corpus))
            stack.enter_context(patch.object(w, 'collect_training', return_value=(toy_data(), w.training_summary(toy_data()), {})))
            fit = stack.enter_context(patch.object(w, 'fit_event_ranker'))
            predict = stack.enter_context(patch.object(w, 'predict'))
            result = w.assay_job(SimpleNamespace(), run, folder)
            self.assertEqual(result['next_action'], 'insufficient_supervision_no_submission')
            self.assertFalse(result['model_trained'])
            fit.assert_not_called()
            predict.assert_not_called()

    def test_workflow_freezes_every_prediction_before_gt_and_never_edits(self):
        corpus, events = fixture_corpus(), []
        def fit(*args):
            events.append(('fit', None))
            return linear()
        def predict(folder, corpus, name, model, comparator, calibration=False):
            self.assertEqual(sum(e[0] == 'fit' for e in events), 2)
            if not calibration:
                self.assertTrue((run/'models.json').is_file())
            events.append(('predict', name))
            t = table()
            return {k: t[k] for k in ('ids', 'scores', 'changed')}
        def annotate(folder, corpus, name, t):
            phase = 'calibration' if name.endswith('_cal') else 'development'
            self.assertTrue((run/(phase+'-prediction-manifest.json')).is_file())
            needed = [n for g in corpus['splits'].values() for n in g['calibration']] if phase == 'calibration' else corpus['development']
            self.assertTrue(all(('predict', n) in events for n in needed))
            return {**t, 'labels': table()['labels'], 'fork_ranks': []}, 'gt'
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run, folder = Path(d), Path(d)/'source'
            folder.mkdir()
            for group in corpus['splits']:
                dump(folder/('ranker-'+group+'.json'), dict(ranker=linear(), margin=None))
            stack.enter_context(patch.object(w, 'verify_sources', return_value=corpus))
            stack.enter_context(patch.object(w, 'collect_training', return_value=(toy_data(), {}, {})))
            stack.enter_context(patch.object(w, 'sufficient', return_value=True))
            stack.enter_context(patch.object(w, 'fit_event_ranker', side_effect=fit))
            stack.enter_context(patch.object(w, 'predict', side_effect=predict))
            stack.enter_context(patch.object(w, 'evaluate_prediction', side_effect=annotate))
            stack.enter_context(patch.object(w.source, 'tree_fingerprint', return_value='gt'))
            edits = stack.enter_context(patch.object(w.source, 'apply_transactions', side_effect=AssertionError('No graph edits')))
            result = w.assay_job(SimpleNamespace(), run, folder)
            edits.assert_not_called()
            self.assertEqual(result['next_action'], 'rejected_no_kaggle_submission')
            for k in ('tracking_evaluated', 'graph_exported', 'confirmation_evaluated', 'visible_evaluated', 'kaggle_submission_allowed', 'promoted'):
                self.assertFalse(result[k])
            self.assertFalse(list(run.glob('*.csv')))
            self.assertEqual(len(list(run.glob('*.scores.npz'))), 4)

    def test_cpu_task_and_node_selection_leave_gpu_rules_intact(self):
        from ops.biohub import resource_profile, child_environment, launch_job
        self.assertEqual(resource_profile('learn-events'), 'cpu_evaluation_v1')
        self.assertEqual(child_environment('learn-events')['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(resource_profile('prepare'), 'gpu_exclusive_v1')
        with self.assertRaisesRegex(ValueError, 'A1555'):
            launch_job(SimpleNamespace(task='learn-events', node='A1688'), Path('unused'), [])

    def test_current_full_tests_are_required_and_assay_cannot_be_packaged(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'verify_sources'):
            root = Path(d)/'runs'
            dump(root/'source/receipt.json', dict(status='completed', task='rank-repairs', code_hash='old'))
            args = SimpleNamespace(task='learn-events', artifact_root=d, source_run='source',
                tests_run='tests', code_hash='current', node='A1555')
            result = dict(tests_passed=True, tests_run=250, tests_expected=250,
                suite='full_discovery', skipped=0, errors=0, failures=0)
            receipt = dict(task='tests', status='completed', code_hash='current', node='A1555', result=result)
            dump(root/'tests/receipt.json', receipt)
            self.assertEqual(source_run(args), root/'source')
            for bad in ({**result, 'skipped': 1}, {**result, 'tests_expected': 300}, {**result, 'suite': 'local_cpu'}):
                dump(root/'tests/receipt.json', {**receipt, 'result': bad})
                with self.assertRaises(ValueError):
                    source_run(args)
            dump(root/'assay/receipt.json', dict(task='learn-events', status='completed', code_hash='current'))
            args.source_run, args.task = 'assay', 'package'
            with self.assertRaisesRegex(ValueError, 'Expected source task experiment'):
                source_run(args)
