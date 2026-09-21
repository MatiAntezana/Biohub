"""014 toy-only tests. No real images, annotations, detector or fit executed."""
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

from core import coverage_audit as w, supervision_audit as a
from core.artifacts import dump, digest
from test_mitosis_selector import graph, ctx, ToyImage
from test_event_ranking import annotation


def vectors(values):
    x = np.zeros((len(values), len(a.features.NAMES)))
    x[:, 0] = values
    return x


def fixtures():
    cfg = w.config()
    names = sorted(cfg['exposed_training_only'])
    historical = dict(data_root='fake-data', development=['44b6_dev'],
        splits={g: dict(fit=[g+'_fit'], calibration=[g+'_cal']) for g in ('44b6', '6bba')})
    plan = dict(all_datasets=names+['44b6_dev', '44b6_confirm', 'visible'],
        public_info={n: dict(divisions=cfg['exposed_training_only'][n]) for n in names},
        panels=dict(development=['44b6_dev'], confirmation=['44b6_confirm'], visible=['visible']),
        data_root='fake-data')
    return cfg, names, historical, plan


class NoveltyAuditTests(unittest.TestCase):
    def test_signed_equivalence_and_opposite_labels(self):
        prior = a.signatures(vectors([-2]), [-1], ['v:1'])
        keep, report = a.deduplicate(vectors([2, 2]), [1, -1], ['v:1', 'v:1'], prior)
        self.assertEqual(keep.tolist(), [1])
        self.assertEqual(report['counts'], dict(duplicate_prior=1, novel=1))

    def test_repeated_rows_are_one_comparison_not_new_events(self):
        keep, report = a.deduplicate(vectors([1, 1, 2]), [1]*3, ['v:1']*3)
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(report['novel_mothers'], 1)
        self.assertEqual(report['novel_volumes'], 1)

    def test_different_mothers_and_datasets_not_merged(self):
        keep, report = a.deduplicate(vectors([1]*3), [1]*3, ['v:1', 'v:2', 'other:1'])
        self.assertEqual(len(keep), 3)
        self.assertEqual(report['novel_volumes'], 2)

    def test_zero_signed_zero_and_near_values(self):
        self.assertEqual(a.signatures(vectors([0.]), [1], ['v:1']),
                         a.signatures(vectors([-0.]), [-1], ['v:1']))
        keep, report = a.deduplicate(vectors([0., 1., 1.+1e-12]), [1]*3, ['v:1']*3)
        self.assertEqual(keep.tolist(), [1, 2])
        self.assertEqual(report['counts']['zero_information'], 1)

    def test_invalid_vectors_labels_keys_rejected(self):
        for x, y, k in ((vectors([np.nan]), [1], ['v:1']), (vectors([1]), [0], ['v:1']),
                        (vectors([1]), [1], ['missing']), (np.zeros((1, 2)), [1], ['v:1']),
                        (vectors([1]), [1, 1], ['v:1'])):
            with self.assertRaises(ValueError):
                a.signatures(x, y, k)

    def test_empty_supervision_is_valid_audit_not_permission_to_fit(self):
        keep, report = a.deduplicate(vectors([]), [], [])
        self.assertEqual(keep.tolist(), [])
        self.assertEqual(report['novel_mothers'], 0)


class CoverageGeometryTests(unittest.TestCase):
    def report(self, mapping=None, raw=None, forks=None, cfg=None, triples=None, c=None):
        c = ctx() if c is None else c
        g = c['anchor']
        m, children, parents = annotation()
        return a.coverage('fixture', g, g.coords if raw is None else raw,
            np.array([[0, 2, 3]]) if forks is None else forks, c,
            m if mapping is None else mapping, children, parents,
            {100: g.coords[[0, 2, 3]]} if triples is None else triples,
            dict(matching_radius_um=7., radius_um=12., max_children=6) if cfg is None else cfg)

    def test_covered_fork_has_known_contrast_and_no_edits(self):
        g = graph()
        before = g.edges.copy()
        result = self.report(c=ctx(g))
        self.assertEqual(result['stages'], {'usable_event': 1})
        self.assertEqual(result['endpoint_categories'], {'matched': 3})
        self.assertFalse(result['graph_modified'])
        np.testing.assert_array_equal(g.edges, before)

    def test_near_unmatched_anchor_not_called_detector_miss(self):
        mapping, _, _ = annotation()
        del mapping[103]
        result = self.report(mapping=mapping)
        row = result['events'][0]['endpoints'][2]
        self.assertEqual(result['stages'], {'unmatched_endpoint': 1})
        self.assertEqual(row['category'], 'near_anchor_unmatched')
        self.assertEqual(row['anchor']['nearest_id'], 3)

    def test_covered_but_unknown_displaced_parent_is_not_usable(self):
        mapping, _, _ = annotation()
        del mapping[101]
        result = self.report(mapping=mapping)
        self.assertEqual(result['stages'], {'covered_without_confirmed_contrast': 1})
        self.assertEqual(result['events'][0]['unknown_reasons']['unmapped_parent_detection'], 1)

    def test_raw_near_anchor_absent_is_not_exact_pruning_attribution(self):
        g = graph()
        point = np.array([1., 30, 30, 30])
        row = a.endpoint(99, point, {}, g.coords, np.array([point]), np.ones(3), 7.)
        self.assertEqual(row['category'], 'near_raw_but_no_near_anchor')
        self.assertIsNone(row['matched_anchor_id'])

    def test_no_raw_or_anchor_near_only_refers_to_saved_candidates(self):
        g = graph()
        row = a.endpoint(99, np.array([1., 30, 30, 30]), {}, g.coords, g.coords, np.ones(3), 7.)
        self.assertEqual(row['category'], 'no_near_detection_in_saved_candidates')

    def test_nearest_same_time_only_and_ties_preserved(self):
        coords = np.array([[0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 2]], float)
        result = a.nearest(coords, np.array([1, 1, 1, 1]), np.ones(3), 1.)
        self.assertEqual(result['nearest_ties'], [1, 2])
        self.assertEqual(result['within_radius'], 2)
        self.assertIsNone(a.nearest(coords, np.array([2, 1, 1, 1]), np.ones(3), 1.)['nearest_id'])

    def test_axial_distance_uses_physical_scale(self):
        result = a.nearest(np.array([[0, 2, 0, 0]]), np.zeros(4), np.array([2., .5, .5]), 3.)
        self.assertEqual(result['distance_um'], 4.)
        self.assertEqual(result['within_radius'], 0)

    def test_outside_radius_explains_missing_fork(self):
        result = self.report(forks=np.empty((0, 3), int),
            cfg=dict(matching_radius_um=7., radius_um=.5, max_children=6))
        self.assertEqual(result['stages'], {'outside_frozen_pool': 1})
        self.assertTrue(all(x['outside_radius'] for x in result['events'][0]['candidate_limits']))

    def test_top_k_and_ties_use_same_id_order_as_frozen_enumerator(self):
        result = self.report(forks=np.empty((0, 3), int),
            cfg=dict(matching_radius_um=7., radius_um=12., max_children=1))
        self.assertEqual([x['rank_all_next_frame'] for x in result['events'][0]['candidate_limits']], [1, 2])
        self.assertTrue(result['events'][0]['candidate_limits'][1]['outside_top_k'])

    def test_missing_fork_inside_frozen_envelope_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not explained'):
            self.report(forks=np.empty((0, 3), int))

    def test_shared_unknown_changes_cancel(self):
        good = dict(add=[(1, 2), (8, 9)], remove=[])
        other = dict(add=[(3, 2), (8, 9)], remove=[])
        reasons = a.unknown_reasons(good, other, {101: 1, 102: 2, 103: 3}, {102: 101})
        self.assertEqual(reasons, {})

    def test_unknown_parent_and_endpoint_reasons_do_not_make_negatives(self):
        reasons = a.unknown_reasons(dict(add=[(0, 1), (2, 3)], remove=[]),
            dict(add=[], remove=[]), {100: 0, 101: 1}, {})
        self.assertEqual(reasons, {'child_gt_parent_unannotated': 1,
            'unmapped_parent_detection': 1, 'unmapped_child_detection': 1})

    def test_noninjective_matching_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-injective'):
            self.report(mapping={100: 0, 102: 0, 103: 3})

    def test_invalid_raw_coordinates_rejected(self):
        for raw in (np.array([[0, np.nan, 0, 0]]), np.array([[.5, 0, 0, 0]]), np.zeros((1, 3))):
            with self.assertRaises(ValueError):
                self.report(raw=raw)

    def test_frame_mismatch_rejected(self):
        triples = {100: graph().coords[[0, 2, 3]].copy()}
        triples[100][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'frame mismatch'):
            self.report(triples=triples)

    def test_coverage_accounting_must_agree_with_frozen_extraction(self):
        result = self.report()
        m, children, parents = annotation()
        _, _, stats = w.old.event_examples('fixture', np.array([[0, 2, 3]]), ctx(), m, children, parents)
        w.verify_dataset_result(result, stats)
        with self.assertRaises(ValueError):
            w.verify_dataset_result(result, {**stats, 'usable_events': 2})


class CoverageWorkflowTests(unittest.TestCase):
    def test_only_three_approved_exposed_training_samples(self):
        cfg, names, corpus, plan = fixtures()
        self.assertEqual(w.expansion_names(plan, corpus, cfg), names)
        self.assertEqual(sum(cfg['exposed_training_only'].values()), 5)

    def test_reserved_historical_and_undeclared_exposure_rejected(self):
        cfg, names, corpus, plan = fixtures()
        for key in ('development', 'confirmation', 'visible'):
            changed = {**plan, 'panels': {**plan['panels'], key: [names[0]]}}
            with self.assertRaises(ValueError):
                w.expansion_names(changed, corpus, cfg)
        corpus['splits']['44b6']['calibration'].append(names[0])
        with self.assertRaises(ValueError):
            w.expansion_names(plan, corpus, cfg)
        with self.assertRaises(ValueError):
            w.expansion_names(plan, fixtures()[2], {**cfg, 'exposed_training_only': {'other': 1}})

    def test_count_drift_refused(self):
        cfg, names, corpus, plan = fixtures()
        plan['public_info'][names[0]]['divisions'] = 17
        with self.assertRaises(ValueError):
            w.expansion_names(plan, corpus, cfg)

    def test_manifest_preserves_all_historical_science(self):
        archived = {'core/production.py': 'frozen', 'ops/biohub.py': 'old'}
        cfg = {**w.config(), 'source_release': hashlib.sha256(json.dumps(archived, sort_keys=True).encode()).hexdigest()}
        current = {**archived, **dict.fromkeys(w.ADDITIONS, 'new'), 'ops/biohub.py': 'new'}
        with patch.object(w, 'config', return_value=cfg):
            self.assertEqual(w.parent_manifest(current, archived), archived)
            for bad in ({**current, 'core/production.py': 'changed'}, {**current, 'new.py': 'extra'}):
                with self.assertRaises(ValueError):
                    w.parent_manifest(bad, archived)

    def test_wrong_source_ids_fail_before_io(self):
        for fn in (w.verify_sources, w.verify_corpus):
            with self.assertRaises(ValueError):
                fn(Path('wrong'), {})

    def test_missing_source_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            w.audit_inputs(Path(d), Path(d)/'plan.json')

    def test_task_cpu_gpu_separation_and_new_node_guard(self):
        from ops.biohub import resource_profile, CPU_PROFILE, GPU_PROFILE, launch_job
        self.assertEqual(resource_profile('audit-coverage'), CPU_PROFILE)
        self.assertEqual(resource_profile('prepare-coverage'), GPU_PROFILE)
        for task in ('audit-coverage', 'prepare-coverage'):
            with patch('ops.biohub.doctor') as doctor, self.assertRaises(ValueError):
                launch_job(SimpleNamespace(task=task, node='A1688'), Path('unused'), [])
            doctor.assert_not_called()

    def test_occupied_gpu_still_blocks_preparation_and_requires24_gib(self):
        from ops.biohub import resource_admission
        env = dict(disk_free_gib=30, gpu='RTX 4090, 22000 MiB, 100 %', gpu_processes='foreign',
                   cpu=dict(mem_available_gib=40, mem_total_gib=64, load_1m=0, host_logical_cpus=16, allowed_cpus=[0, 1]))
        self.assertFalse(resource_admission('prepare-coverage', env)['allowed'])
        self.assertTrue(resource_admission('audit-coverage', env)['allowed'])
        env.update(gpu_processes='', disk_free_gib=23)
        self.assertFalse(resource_admission('prepare-coverage', env)['allowed'])

    def test_tests_must_be_current_full_on_same_node(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'verify_sources'):
            root = Path(d)/'runs'
            dump(root/'source/receipt.json', {})
            args = SimpleNamespace(artifact_root=d, source_run='source', tests_run='tests',
                                   task='prepare-coverage', code_hash='current', node='A1555')
            result = dict(tests_passed=True, suite='full_discovery', skipped=0, errors=0, failures=0,
                          tests_run=300, tests_expected=300)
            receipt = dict(task='tests', status='completed', code_hash='current', node='A1555', result=result)
            dump(root/'tests/receipt.json', receipt)
            self.assertEqual(source_run(args), root/'source')
            for bad in ({**receipt, 'node': 'A1688'}, {**receipt, 'code_hash': 'old'},
                        {**receipt, 'result': {**result, 'tests_run': 247, 'tests_expected': 247}},
                        {**receipt, 'result': {**result, 'skipped': 1}}):
                dump(root/'tests/receipt.json', bad)
                with self.assertRaises(ValueError):
                    source_run(args)

    def test_diagnostic_cannot_be_packaged(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as d:
            dump(Path(d)/'runs/audit/receipt.json', dict(status='completed', task='audit-coverage', code_hash='current'))
            args = SimpleNamespace(artifact_root=d, source_run='audit', task='package', code_hash='current')
            with self.assertRaisesRegex(ValueError, 'Expected source task experiment'):
                source_run(args)

    def test_preparation_uses_only_frozen_images_and_no_gt(self):
        cfg, names, historical, plan = fixtures()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run = Path(d)
            stack.enter_context(patch.object(w, 'verify_sources', return_value=({}, historical, plan)))
            stack.enter_context(patch.object(w.shutil, 'disk_usage', return_value=SimpleNamespace(free=30*1024**3)))
            cached = stack.enter_context(patch('ops.workflow.cache_panel'))
            stack.enter_context(patch.object(w, 'load_reference', return_value={n: graph() for n in names}))
            stack.enter_context(patch('core.nextgen.images.ImageSource', return_value=ToyImage()))
            stack.enter_context(patch.object(w, 'payload_files', return_value={'fixture': 'hash'}))
            matched = stack.enter_context(patch.object(w.old.source, 'match_annotation'))
            result = w.prepare_job(SimpleNamespace(artifact_root='remote'), run, Path(cfg['source_run']))
            self.assertEqual(cached.call_args.args[0], names)
            self.assertEqual(cached.call_args.args[1], 'fake-data')
            matched.assert_not_called()
            self.assertTrue(result['ready_for_coverage_audit'])
            self.assertFalse(result['gt_opened'])
            self.assertTrue(all(result[k] is False for k in w.FALSE_FLAGS))
            corpus = json.loads((run/'corpus.json').read_text())
            self.assertEqual(corpus['splits'], {'44b6': {'fit': names, 'calibration': []}})

    def test_preparation_source_drift_fails_after_image_processing(self):
        cfg, names, historical, plan = fixtures()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            stack.enter_context(patch.object(w, 'verify_sources', side_effect=[({}, historical, plan), ({'drift': True}, historical, plan)]))
            stack.enter_context(patch.object(w.shutil, 'disk_usage', return_value=SimpleNamespace(free=30*1024**3)))
            stack.enter_context(patch('ops.workflow.cache_panel'))
            stack.enter_context(patch.object(w, 'load_reference', return_value={n: graph() for n in names}))
            stack.enter_context(patch('core.nextgen.images.ImageSource', return_value=ToyImage()))
            with self.assertRaisesRegex(ValueError, 'source changed'):
                w.prepare_job(SimpleNamespace(artifact_root='remote'), Path(d), Path(cfg['source_run']))

    def test_no_fit_score_or_graph_apply_calls_in014(self):
        text = inspect.getsource(w) + inspect.getsource(a)
        for forbidden in ('fit_ranker(', 'fit_event_ranker(', 'apply_transactions(', 'official_metrics(',
                          'evaluate_panel(', 'score_universe('):
            self.assertNotIn(forbidden, text)

    def test_dataset_gt_fingerprint_drift_rejected_before_matching(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            run = Path(d)
            raw = run/'cache-training/raw'
            raw.mkdir(parents=True)
            with (raw/'fixture_candidates.npz').open('xb') as h:
                np.savez(h, coords=graph().coords)
            stack.enter_context(patch.object(w.old.source, 'open_evidence', return_value=(graph(), None, [], ctx())))
            stack.enter_context(patch.object(w.old.source, 'tree_fingerprint', return_value='changed'))
            matched = stack.enter_context(patch.object(w.old.source, 'match_annotation'))
            with self.assertRaisesRegex(ValueError, 'GT changed'):
                w.dataset_audit(run, dict(data_root='fake'), 'fixture', 'original', run, w.config())
            matched.assert_not_called()

    def test_full_cpu_audit_toy_pipeline_deduplicates_without_opening_reserved_panels(self):
        cfg, names, historical, plan = fixtures()
        cfg = {**cfg, 'exposed_training_only': dict.fromkeys(names, 1)}
        expanded = dict(data_root='fake-data', splits={'44b6': dict(fit=names, calibration=[])}, datasets={})
        historical['datasets'] = {}
        mapping, children, parents = annotation()
        annotation_result = (mapping, children, parents, {100: graph().coords[[0, 2, 3]]})
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            runs = Path(d)
            folder, run = runs/cfg['prepare_run'], runs/'audit'
            folder.mkdir(); run.mkdir()
            before = dict(historical_novelty={})
            stack.enter_context(patch.object(w, 'config', return_value=cfg))
            stack.enter_context(patch.object(w, 'verify_corpus', return_value=(before, historical, expanded, plan)))
            stack.enter_context(patch.object(w.old.source, 'tree_fingerprint', return_value='gt'))
            matched = stack.enter_context(patch.object(w.old.source, 'match_annotation', return_value=annotation_result))
            fit = stack.enter_context(patch.object(w.old.old, 'fit_ranker', side_effect=AssertionError('No fitting')))
            for corpus, directory in ((historical, runs/cfg['corpus_run']), (expanded, folder)):
                raw = directory/'cache-training/raw'
                raw.mkdir(parents=True)
                for group, split in corpus['splits'].items():
                    for name in split['fit']:
                        corpus['datasets'][name] = w.old.source.save_evidence(directory/(name+'.evidence.npz'),
                            graph(), ToyImage(), w.old.source.config())
                        corpus['datasets'][name]['group'] = group
                        with (raw/(name+'_candidates.npz')).open('xb') as h:
                            np.savez(h, coords=graph().coords)
            for group in historical['splits']:
                name = historical['splits'][group]['fit'][0]
                _, _, forks, context = w.old.source.open_evidence(runs/cfg['corpus_run'], name, historical)
                x, rows, stats = w.old.event_examples(name, forks, context, mapping, children, parents)
                dump(runs/cfg['source_run']/('training-'+group+'.json'),
                     dict(datasets={name: stats}, gt_sha256={name: 'gt'}))
                with (runs/cfg['source_run']/('training-'+group+'.npz')).open('xb') as h:
                    np.savez(h, x=x, keys=np.array([r['event'] for r in rows]), strata=np.array(['event']*len(x)))
            direct = runs/w.old.config()['source_run']
            direct.mkdir()
            with (direct/'preferences-44b6.npz').open('xb') as h:
                np.savez(h, differences=vectors([]), signs=np.array([]), mother_keys=np.array([], str))
            result = w.audit_job(SimpleNamespace(), run, folder)
            self.assertEqual(result['dataset_count'], 5)
            self.assertEqual(result['new_usable_biological_forks'], 3)
            self.assertTrue(result['source_unchanged'])
            # This fully annotated toy universe really has six new pairwise
            # differences plus duplicate KEEP comparisons. Do not force the
            # historical013 zero-novelty finding onto new evidence.
            self.assertEqual(result['new_events_vs012_and_new_direct']['novel_comparisons'], 6)
            self.assertGreater(result['new_events_vs012_and_new_direct']['counts']['duplicate_prior'], 0)
            self.assertGreater(result['new_unique_preferences'], 0)
            self.assertEqual(result['next_action'], 'designer_review_no_submission')
            self.assertTrue(all(result[k] is False for k in w.FALSE_FLAGS))
            self.assertFalse(result['graph_exported'])
            fit.assert_not_called()
            expected = set(names+['44b6_fit', '6bba_fit'])
            self.assertEqual({c.args[1].stem for c in matched.call_args_list}, expected)
            with np.load(run/'new-supervision-44b6.npz') as f:
                sig = a.signatures(f['signed_x'], np.ones(len(f['keys'])), f['keys'])
                self.assertEqual(len(sig), len(set(sig)))

    def test_prepared_corpus_verifier_rejects_payload_and_partition_drift(self):
        cfg, names, historical, plan = fixtures()
        manifest = {'test': 'fixture'}
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            folder = Path(d)/cfg['prepare_run']
            folder.mkdir()
            stack.enter_context(patch.object(w, 'verify_sources', return_value=({}, historical, plan)))
            payload = stack.enter_context(patch.object(w, 'payload_files', return_value={'file': 'hash'}))
            corpus = dict(schema='exposed_training_corpus_014', disposition=w.PREP_SCOPE, code_hash=identity,
                source_run=cfg['source_run'], recipe=cfg, data_root=plan['data_root'],
                splits={'44b6': dict(fit=names, calibration=[])}, previous_exposure_declared=True,
                training_only=True, source_unchanged=True, gt_opened=False, graph_edits_applied=False,
                payload_sha256={'file': 'hash'}, datasets={}, **dict.fromkeys(w.FALSE_FLAGS, False))
            for name in names:
                dump(folder/(name+'.evidence.npz'), {'fixture': name})
                corpus['datasets'][name] = dict(group='44b6', panel='previously_exposed_training_only',
                    sha256=digest(folder/(name+'.evidence.npz')))
            receipt = dict(task='prepare-coverage', status='completed', code_hash=identity, code_manifest=manifest,
                node='A1555', source_run=cfg['source_run'], result=dict(ready_for_coverage_audit=True))
            def write(c):
                dump(folder/'corpus.json', c)
                dump(folder/'receipt.json', {**receipt, 'result': {**receipt['result'],
                    'corpus_sha256': digest(folder/'corpus.json')}})
            write(corpus)
            self.assertEqual(w.verify_corpus(folder, manifest)[2], corpus)
            for bad in ({**corpus, 'gt_opened': True}, {**corpus, 'confirmation_evaluated': True},
                        {**corpus, 'training_only': False}, {**corpus, 'previous_exposure_declared': False},
                        {**corpus, 'splits': {'44b6': {'fit': names, 'calibration': ['reserved']}}}):
                write(bad)
                with self.assertRaises(ValueError):
                    w.verify_corpus(folder, manifest)
            write(corpus)
            payload.return_value = {'file': 'changed'}
            with self.assertRaises(ValueError):
                w.verify_corpus(folder, manifest)
