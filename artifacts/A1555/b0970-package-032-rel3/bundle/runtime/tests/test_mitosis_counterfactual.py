"""Counterfactual evaluation is quarantined from production and source graphs."""
import hashlib
import json
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.artifacts import clean, dump
from core.mitosis_counterfactual import (ADDITIONS, DRIVERS, SCOPE, SCENARIOS,
    config, verify_sources, oracle_copy, audited_forks, scenario_forks,
    assert_baseline, compare_rows, counterfactual_job, metric_row)
from core.mitosis_proposals import edit_footprint, topology_maps
from core.nextgen.contracts import Lineage


def fixture():
    return Lineage(np.array([[0, 10, 10, 10], [0, 10, 10, 14],
        [1, 10, 10, 9], [1, 10, 10, 11], [1, 10, 10, 15], [1, 10, 10, 17]], float),
        np.array([[0, 2], [1, 3]], np.int64)).validate()


def fork(anchor, ids=(0, 2, 3), *, covered=True, rank=1):
    return dict(gt_ids=[10, 12, 13], anchor_ids=list(ids), covered=covered,
        reason="scored" if covered else "outside_radius", rank_worst=rank,
        edit_footprint=clean(edit_footprint(ids, *topology_maps(anchor))))


def row(name, graph):
    # Unit fixture only: real metric parity is a mandatory remote runtime test.
    count = len(graph.edges)
    return dict(dataset=name, group=name.split("_", 1)[0], split="development",
        counts=dict(edge_tp=count, edge_fp=0, edge_fn=0), edge_tp=count,
        edge_fp=0, edge_fn=0, num_pred_nodes=len(graph.coords),
        node_recall=1., total_node_ratio=0., edge_jaccard=1., adj_edge_jaccard=1.)


class MitosisCounterfactualTests(unittest.TestCase):
    def test_reassignment_is_atomic_preserves_coordinates_and_other_edges(self):
        anchor = fixture()
        c, e = anchor.coords.copy(), anchor.edges.copy()
        result, report = oracle_copy(anchor, [[0, 2, 3]])
        self.assertEqual(set(map(tuple, result.edges)), {(0, 2), (0, 3)})
        self.assertEqual(report["removed_edges"], [(1, 3)])
        self.assertEqual(report["added_edges"], [(0, 3)])
        self.assertEqual(report["edge_count_delta"], 0)
        np.testing.assert_array_equal(anchor.coords, c)
        np.testing.assert_array_equal(anchor.edges, e)
        np.testing.assert_array_equal(result.coords, c)
        self.assertFalse(np.shares_memory(anchor.coords, result.coords))
        self.assertFalse(np.shares_memory(anchor.edges, result.edges))

    def test_additive_noop_and_empty_transactions(self):
        anchor = fixture()
        result, report = oracle_copy(anchor, [[0, 2, 4]])
        self.assertEqual(report["edge_count_delta"], 1)
        self.assertIn((1, 3), set(map(tuple, result.edges)))
        for requests in ([], [[0, 2, 4]]):
            copy, edits = oracle_copy(result, requests)
            np.testing.assert_array_equal(copy.edges, result.edges)
            self.assertEqual(edits["added_edges"], [])
            self.assertEqual(edits["removed_edges"], [])

    def test_simultaneous_conflicting_daughters_or_mothers_reject_without_mutation(self):
        anchor = fixture()
        before = anchor.edges.copy()
        for requests in ([[0, 2, 3], [1, 3, 4]], [[0, 2, 3], [0, 4, 5]]):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                oracle_copy(anchor, requests)
        np.testing.assert_array_equal(before, anchor.edges)

    def test_simultaneous_requests_do_not_depend_on_order(self):
        anchor = fixture()
        a, _ = oracle_copy(anchor, [[0, 2, 3], [1, 4, 5]])
        b, _ = oracle_copy(anchor, [[1, 4, 5], [0, 2, 3]])
        np.testing.assert_array_equal(a.edges, b.edges)
        self.assertEqual(set(map(tuple, a.edges)), {(0, 2), (0, 3), (1, 4), (1, 5)})

    def test_invalid_ids_times_and_duplicate_daughter_rejected(self):
        for requests in ([[0, 2, 2]], [[0, 1, 2]], [[-1, 2, 3]], [[9, 2, 3]],
                         [[0., 2., 3.]], [[0, 2]]):
            with self.assertRaises(ValueError):
                oracle_copy(fixture(), requests)

    def test_scenarios_exclude_present_unmatched_and_top1_ties(self):
        anchor = fixture()
        a = fork(anchor)
        b = fork(anchor, rank=2)
        c = fork(anchor, covered=False)
        present = {**a, "edit_footprint": {"kind": "already_present"}}
        missing = {**a, "anchor_ids": [0, 2, None]}
        gap = {**a, "reason": "nonconsecutive_gt_fork"}
        result = scenario_forks([a, b, c, present, missing, gap])
        self.assertEqual(result["covered_nonpresent"], [a, b])
        self.assertEqual(result["covered_unique_top1_nonpresent"], [a])
        self.assertEqual(result["matched_nonpresent"], [a, b, c])

    def test_fresh_matching_and_gt_identity_required(self):
        anchor = fixture()
        info = dict(gt_sha256="abc", scale_um=[1, 1, 1], exact_gt_forks=[fork(anchor)])
        matching = ({10: 0, 12: 2, 13: 3}, {10: {12, 13}}, {}, {10: anchor.coords[[0, 2, 3]].copy()})
        with patch("core.mitosis_counterfactual.tree_fingerprint", return_value="abc"), patch("core.mitosis_counterfactual.match_annotation", return_value=matching):
            result = audited_forks(anchor, info, Path("fake"))
            self.assertEqual(result[0]["endpoint_residual_um"], [0., 0., 0.])
            bad = {**info, "exact_gt_forks": [{**fork(anchor), "anchor_ids": [1, 2, 3]}]}
            with self.assertRaisesRegex(ValueError, "Full matching"):
                audited_forks(anchor, bad, Path("fake"))
            bad = {**info, "exact_gt_forks": [{**fork(anchor), "edit_footprint": {}}]}
            with self.assertRaisesRegex(ValueError, "footprint"):
                audited_forks(anchor, bad, Path("fake"))
        with patch("core.mitosis_counterfactual.tree_fingerprint", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "GT differs"):
                audited_forks(anchor, info, Path("fake"))

    def test_baseline_count_and_float_parity_cannot_be_waived_by_split_label(self):
        actual = row("44b6_dev", fixture())
        archived = {**actual, "split": "confirmation"}
        assert_baseline(actual, archived, 1e-10)
        for bad in ({**archived, "counts": {}}, {**archived, "adj_edge_jaccard": .9},
                    {**archived, "node_recall": float("nan")}, {**archived, "dataset": "6bba_other"}):
            with self.assertRaises(ValueError):
                assert_baseline(actual, bad, 1e-10)

    def test_source_chain_rejects_scientific_changes_and_bad_pins(self):
        cfg = {**config(), "parent_science_sha256": hashlib.sha256(b"{}").hexdigest()}
        manifest = dict.fromkeys(ADDITIONS | DRIVERS, "new")
        with patch("core.mitosis_counterfactual.config", return_value=cfg), patch("core.mitosis_counterfactual.verify_proposal_sources"), patch("core.mitosis_counterfactual.verify_file", side_effect=ValueError("bad pin")):
            with self.assertRaisesRegex(ValueError, "bad pin"):
                verify_sources(Path("b0970-prepare-005"), manifest)
            with self.assertRaisesRegex(ValueError, "exact 009"):
                verify_sources(Path("b0970-prepare-005"), {**manifest, "core/new.py": "drift"})
            with self.assertRaisesRegex(ValueError, "exact 009"):
                verify_sources(Path("b0970-proposals-009"), manifest)

    def test_aggregate_rejects_missing_samples_undefined_values_and_node_changes(self):
        def summarise(rows):
            return dict(score=1., edge_jaccard=1., adj_edge_jaccard=1., division_jaccard=1.)
        a = {"44b6_dev": row("44b6_dev", fixture())}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compare_rows(a, {})
        with patch("core.mitosis_counterfactual.official_metrics", return_value=SimpleNamespace(summarise=summarise, COUNT_COLUMNS=("edge_tp",))):
            result = compare_rows(a, a)
            self.assertFalse(result["panel"]["promoted"])
            b = {"44b6_dev": {**a["44b6_dev"], "num_pred_nodes": 99}}
            with self.assertRaisesRegex(ValueError, "cardinality"):
                compare_rows(a, b)
        with patch("core.mitosis_counterfactual.official_metrics", return_value=SimpleNamespace(summarise=lambda rows: dict.fromkeys(("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard"), float("nan")))):
            with self.assertRaisesRegex(ValueError, "Undefined"):
                compare_rows(a, a)

    def test_job_reproduces_all_baselines_before_edits_and_exports_only_json(self):
        names = ["44b6_dev", "6bba_dev"]
        anchors = {name: fixture() for name in names}
        infos = {name: dict(gt_sha256="abc", scale_um=[1.625, .40625, .40625],
            evaluated_data_group=name[:4], shape=[3, 30, 30, 30], exact_gt_forks=[fork(anchors[name])]) for name in names}
        archive = dict(rows=dict(reference=[row(n, anchors[n]) for n in names]))
        events = []
        def measure(graph, r):
            events.append(("metric", r["name"]))
            return row(r["name"], graph)
        def audit(anchor, info, path):
            events.append(("audit", str(path)))
            self.assertEqual(events[:2], [("metric", n) for n in names])
            return info["exact_gt_forks"]
        with tempfile.TemporaryDirectory() as folder:
            source, run = Path(folder)/"b0970-prepare-005", Path(folder)/"new"
            dump(source/"plan.json", dict(data_root="fake", panels=dict(development=names)))
            with patch("core.mitosis_counterfactual.verify_sources", return_value=(dict(datasets=infos), archive)), patch("core.mitosis_counterfactual.PROTOCOL", dict(development=names)), patch("core.mitosis_counterfactual.load_reference", return_value=anchors), patch("core.mitosis_counterfactual.tree_fingerprint", return_value="abc"), patch("core.mitosis_counterfactual.metric_row", side_effect=measure), patch("core.mitosis_counterfactual.audited_forks", side_effect=audit), patch("core.mitosis_counterfactual.compare_rows", return_value={"toy": True}):
                result = counterfactual_job(SimpleNamespace(), run, source)
            self.assertEqual(result["disposition"], SCOPE)
            self.assertTrue(result["uses_gt_for_edits"])
            self.assertTrue(result["tracking_evaluated"])
            for flag in ("confirmation_evaluated", "graph_exported", "source_graph_modified", "promoted", "kaggle_submission_allowed", "deployable_policy_evaluated"):
                self.assertFalse(result[flag])
            self.assertTrue(all(p.suffix == ".json" for p in run.iterdir()))
            for anchor in anchors.values():
                np.testing.assert_array_equal(anchor.edges, fixture().edges)
            payload = json.loads((run/"counterfactual.json").read_text())
            self.assertEqual(set(payload["scenarios"]), set(SCENARIOS))
            self.assertEqual(len(payload["single_fork_effects"]), 2)

    def test_current_tests_are_required_and_oracle_receipt_cannot_enable_production(self):
        from ops.biohub import source_run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dump(root/"runs/b0970-prepare-005/receipt.json", dict(status="completed", task="prepare", code_hash="old"))
            args = SimpleNamespace(artifact_root=root, source_run="b0970-prepare-005", task="audit-mitosis", tests_run="tests", code_hash="current")
            with patch("core.mitosis_counterfactual.verify_sources"):
                dump(root/"runs/tests/receipt.json", dict(status="completed", task="tests", code_hash="old", result=dict(tests_passed=True)))
                with self.assertRaisesRegex(ValueError, "Current diagnostic"):
                    source_run(args)
                dump(root/"runs/tests/receipt.json", dict(status="completed", task="tests", code_hash="current", result=dict(tests_passed=True)))
                self.assertEqual(source_run(args).name, "b0970-prepare-005")
            args.source_run = "audit"
            dump(root/"runs/audit/receipt.json", dict(status="completed", task="audit-mitosis", code_hash="current", result=dict(disposition=SCOPE)))
            for task in ("prepare", "experiment", "package"):
                args.task = task
                with self.assertRaisesRegex(ValueError, "Expected source task"):
                    source_run(args)


class MitosisCounterfactualRuntimeTests(unittest.TestCase):
    def test_official_adapter_matches_production_evaluator_and_recovers_toy_division(self):
        from core.evaluation import make_graph, metric_root
        from core.nextgen.pipeline import evaluate_pairs
        coords = np.array([[0, 10, 10, 10], [1, 10, 10, 10],
            [2, 10, 10, 5], [2, 10, 10, 15], [3, 10, 10, 4], [3, 10, 10, 16]], float)
        edges = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]], np.int64)
        anchor = Lineage(coords, edges[[0, 1, 3, 4]].copy()).validate()
        repaired, _ = oracle_copy(anchor, [[1, 2, 3]])
        def gt(*a, **kw):
            nodes = {}
            for i, c in enumerate(coords):
                nodes.setdefault(int(c[0]), []).append(dict(id=i, pos=c[1:].tolist()))
            return make_graph(nodes, edges.tolist())
        r = dict(name="44b6_toy", group="44b6", split="development", gt="fake.geff", scale_um=[1, 1, 1])
        with patch("ops.workflow.load_graph", side_effect=gt), patch("geff.GeffMetadata.read", return_value=SimpleNamespace(extra=dict(estimated_number_of_nodes=6))), patch("tracksdata.graph.InMemoryGraph.from_geff", side_effect=gt):
            before, after = metric_row(anchor, r), metric_row(repaired, r)
            paired = evaluate_pairs({r["name"]: repaired}, {r["name"]: anchor}, [r], metric_root())
        self.assertEqual(before, paired["rows"]["reference"][0])
        self.assertEqual(after, paired["rows"]["candidate"][0])
        self.assertEqual(after["edge_tp"], 5)
        self.assertEqual(after["division_tp"], 1)
        self.assertEqual(after["division_fn"], 0)
        self.assertLess(before["edge_tp"], after["edge_tp"])
        self.assertLess(before["division_tp"], after["division_tp"])


class WorkerResourceTests(unittest.TestCase):
    """Operational-only regressions; OS and CUDA mutations are mocked on the Mac."""
    def environment(self, **cpu):
        return dict(disk_free_gib=100., gpu="RTX 4090, 2708 MiB, 100 %",
            gpu_processes="49933, VLLM::EngineCore, 21264 MiB",
            cpu={"mem_available_gib": 32., "mem_total_gib": 64., "load_1m": 2.,
                "host_logical_cpus": 16, "allowed_cpus": list(range(16)), **cpu})

    def test_only_tests_and_counterfactual_are_cpu_tasks(self):
        from ops.biohub import CPU_PROFILE, GPU_PROFILE, GPU_TASKS, resource_profile
        for task in ("tests", "audit-mitosis"):
            self.assertEqual(resource_profile(task), CPU_PROFILE)
        for task in GPU_TASKS:
            self.assertEqual(resource_profile(task), GPU_PROFILE)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resource_profile("training-cpu")

    def test_cpu_admission_allows_busy_gpu_but_gpu_tasks_never_do(self):
        from ops.biohub import resource_admission, GPU_TASKS
        for task in ("tests", "audit-mitosis"):
            self.assertTrue(resource_admission(task, self.environment())["allowed"])
        for task in GPU_TASKS:
            result = resource_admission(task, self.environment())
            self.assertFalse(result["allowed"])
            self.assertTrue(any("GPU occupied" in x for x in result["reasons"]))

    def test_cpu_pressure_or_missing_measurements_blocks_admission(self):
        from ops.biohub import resource_admission
        for cpu in (dict(mem_available_gib=15.), dict(mem_total_gib=256.), dict(load_1m=13.),
                    dict(load_1m=float("nan")), dict(host_logical_cpus=0), dict(allowed_cpus=[]),
                    dict(mem_available_gib=None)):
            self.assertFalse(resource_admission("tests", self.environment(**cpu))["allowed"])
        self.assertFalse(resource_admission("tests", dict(disk_free_gib=100.))["allowed"])

    def test_disk_and_free_gpu_vram_guards_are_preserved(self):
        from ops.biohub import resource_admission
        free = {**self.environment(), "gpu_processes": "", "gpu": "RTX 4090, 24000 MiB, 0 %"}
        self.assertTrue(resource_admission("prepare", free)["allowed"])
        for gpu in ("RTX 4090, 2708 MiB, 0 %", "unknown"):
            self.assertFalse(resource_admission("prepare", {**free, "gpu": gpu})["allowed"])
        for task in ("tests", "prepare"):
            self.assertFalse(resource_admission(task, {**free, "disk_free_gib": 11.})["allowed"])

    def test_child_cpu_environment_is_sealed_without_changing_parent(self):
        from ops.biohub import child_environment, CPU_ENV, CPU_PROFILE, GPU_PROFILE
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "99", "OTHER": "kept"}, clear=True):
            cpu = child_environment("tests")
            self.assertTrue(all(cpu[k] == v for k, v in CPU_ENV.items()))
            self.assertEqual(cpu["BIOHUB_RESOURCE_PROFILE"], CPU_PROFILE)
            self.assertEqual(cpu["OTHER"], "kept")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            gpu = child_environment("prepare")
            self.assertEqual(gpu["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(gpu["BIOHUB_RESOURCE_PROFILE"], GPU_PROFILE)

    def test_cpu_limits_touch_only_current_process_and_respect_lower_inherited_limits(self):
        import resource
        from ops.biohub import configure_cpu_worker, CPU_ENV, CPU_PROFILE
        state = {"affinity": {0, 1, 2, 3}, "nice": 0}
        def affinity(pid, cpus=None):
            self.assertEqual(pid, 0)
            if cpus is not None:
                state["affinity"] = cpus
            return state["affinity"]
        def nice(increment):
            self.assertGreaterEqual(increment, 0)
            state["nice"] += increment
            return state["nice"]
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {**CPU_ENV, "BIOHUB_RESOURCE_PROFILE": CPU_PROFILE}))
            stack.enter_context(patch("ops.biohub.platform.system", return_value="Linux"))
            stack.enter_context(patch("ops.biohub.sys", SimpleNamespace(modules={})))
            stack.enter_context(patch.object(os, "sched_getaffinity", side_effect=affinity, create=True))
            stack.enter_context(patch.object(os, "sched_setaffinity", side_effect=affinity, create=True))
            stack.enter_context(patch("os.nice", side_effect=nice))
            stack.enter_context(patch("resource.getrlimit", return_value=(4*1024**3, resource.RLIM_INFINITY)))
            setter = stack.enter_context(patch("resource.setrlimit"))
            report = configure_cpu_worker("audit-mitosis")
        self.assertEqual(report["logical_cpus"], [2, 3])
        self.assertEqual(report["nice"], 10)
        setter.assert_called_once_with(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))

    def test_cpu_limits_reject_wrong_task_mac_unsealed_env_and_late_imports(self):
        from ops.biohub import configure_cpu_worker, CPU_ENV, CPU_PROFILE
        with self.assertRaisesRegex(RuntimeError, "approved remote"):
            configure_cpu_worker("prepare")
        with patch("ops.biohub.platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError, "approved remote"):
                configure_cpu_worker("tests")
        with patch("ops.biohub.platform.system", return_value="Linux"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                configure_cpu_worker("tests")
        with patch("ops.biohub.platform.system", return_value="Linux"), patch.dict(os.environ, {**CPU_ENV, "BIOHUB_RESOURCE_PROFILE": CPU_PROFILE}), patch("ops.biohub.sys", SimpleNamespace(modules={"torch": object()})):
            with self.assertRaisesRegex(RuntimeError, "precede"):
                configure_cpu_worker("tests")

    def test_cpu_runtime_rejects_any_cuda_visibility_initialization_or_excess_threads(self):
        from ops.biohub import verify_cpu_runtime
        from unittest.mock import Mock
        torch = Mock()
        torch.cuda.is_available.return_value = False
        torch.cuda.device_count.return_value = 0
        torch.cuda.is_initialized.return_value = False
        torch.get_num_threads.return_value = 2
        torch.get_num_interop_threads.return_value = 1
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertEqual(verify_cpu_runtime(torch)["cuda_visible_devices"], 0)
            for method, bad in ((torch.cuda.is_available, True), (torch.cuda.device_count, 1),
                    (torch.cuda.is_initialized, True), (torch.get_num_threads, 3)):
                old = method.return_value
                method.return_value = bad
                with self.assertRaises(RuntimeError):
                    verify_cpu_runtime(torch)
                method.return_value = old
        torch.cuda.mem_get_info.assert_not_called()

    def test_deferred_launch_does_not_consume_run_id_or_start_process(self):
        from ops.biohub import launch_job, resource_admission
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"runs/tests-cpu"
            args = SimpleNamespace(artifact_root=folder, task="tests", run_id="tests-cpu")
            env = self.environment(mem_available_gib=1.)
            env["admission"] = resource_admission("tests", env)
            with patch("ops.biohub.doctor", return_value=env), patch("ops.biohub.subprocess.Popen") as spawn:
                result = launch_job(args, run, ["launch"])
            self.assertEqual(result["status"], "deferred_resources")
            self.assertFalse(run.exists())
            spawn.assert_not_called()

    def test_cpu_launch_seals_env_and_records_profile_without_reusing_id(self):
        from ops.biohub import launch_job, resource_admission, CPU_PROFILE
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"runs/tests-cpu"
            args = SimpleNamespace(artifact_root=folder, task="tests", run_id="tests-cpu", arm="mitosis_add",
                node="A1688", source_run=None, tests_run=None, code_hash="current")
            env = self.environment()
            env["admission"] = resource_admission("tests", env)
            with patch("ops.biohub.doctor", return_value=env), patch("ops.biohub.subprocess.Popen", return_value=SimpleNamespace(pid=123)) as spawn:
                result = launch_job(args, run, ["launch", "--task", "tests"])
                self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                with self.assertRaises(FileExistsError):
                    launch_job(args, run, ["launch"])
                self.assertEqual(spawn.call_count, 1)
            self.assertEqual(result["resource_profile"], CPU_PROFILE)
            self.assertEqual(json.loads((run/"receipt.json").read_text())["resource_profile"], CPU_PROFILE)

    def test_test_outcome_fails_skips_partial_discovery_and_empty_suites(self):
        from ops.verify import test_outcome
        result = SimpleNamespace(wasSuccessful=lambda: True, skipped=[], failures=[], errors=[], testsRun=5)
        self.assertTrue(test_outcome(result, 5, True)["tests_passed"])
        self.assertFalse(test_outcome(result, 6, True)["tests_passed"])
        result.skipped = [("a", "not available")]
        self.assertFalse(test_outcome(result, 5, True)["tests_passed"])
        result.skipped, result.testsRun = [], 0
        self.assertFalse(test_outcome(result, 0, True)["tests_passed"])

    def test_cpu_snapshot_is_readonly_and_converts_kib_to_gib(self):
        from ops.biohub import cpu_snapshot
        with patch("pathlib.Path.read_text", return_value="MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\nOther: 1 kB\n"), patch("os.getloadavg", return_value=(1., 2., 3.)), patch("os.cpu_count", return_value=16), patch.object(os, "sched_getaffinity", return_value={1, 2, 3}, create=True):
            snapshot = cpu_snapshot()
        self.assertEqual(snapshot["mem_available_gib"], 32.)
        self.assertEqual(snapshot["mem_total_gib"], 64.)
        self.assertEqual(snapshot["allowed_cpus"], [1, 2, 3])

    def test_cpu_execute_never_queries_free_vram_and_records_isolation(self):
        from ops.biohub import execute_job, CPU_PROFILE
        from unittest.mock import Mock
        torch = Mock()
        torch.cuda.is_available.return_value = False
        torch.cuda.device_count.return_value = 0
        torch.cuda.is_initialized.return_value = False
        torch.get_num_threads.return_value = 2
        torch.get_num_interop_threads.return_value = 1
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"runs/audit-cpu"
            dump(run/"receipt.json", dict(status="queued", task="audit-mitosis"))
            args = SimpleNamespace(task="audit-mitosis", artifact_root=folder)
            with patch("ops.biohub.configure_cpu_worker", return_value=dict(resource_profile=CPU_PROFILE)), patch("ops.biohub.doctor", return_value=self.environment()), patch("ops.biohub.source_run", return_value=Path("source")), patch("core.mitosis_counterfactual.counterfactual_job", return_value=dict(disposition=SCOPE)) as audit, patch.dict("sys.modules", {"torch": torch}), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
                execute_job(args, run)
            audit.assert_called_once()
            torch.cuda.mem_get_info.assert_not_called()
            self.assertEqual(json.loads((run/"receipt.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((run/"cpu-runtime-after.json").read_text())["cuda_visible_devices"], 0)

    def test_busy_gpu_execute_remains_rejected_before_training(self):
        from ops.biohub import execute_job
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"runs/prepare"
            dump(run/"receipt.json", dict(status="queued", task="prepare"))
            args = SimpleNamespace(task="prepare", artifact_root=folder)
            with patch("ops.biohub.doctor", return_value=self.environment()), patch("ops.workflow.prepare_job") as prepare:
                with self.assertRaisesRegex(RuntimeError, "GPU occupied"):
                    execute_job(args, run)
            prepare.assert_not_called()
            self.assertEqual(json.loads((run/"receipt.json").read_text())["status"], "failed")

    def test_cpu_worker_does_not_bypass_the_shared_biohub_lock(self):
        import fcntl
        from ops.biohub import execute_job
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"runs/audit"
            dump(run/"receipt.json", dict(status="queued", task="audit-mitosis"))
            args = SimpleNamespace(task="audit-mitosis", artifact_root=folder)
            with (Path(folder)/"gpu.lock").open("a") as existing:
                fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch("ops.biohub.configure_cpu_worker", return_value={}), patch("core.mitosis_counterfactual.counterfactual_job") as audit:
                    with self.assertRaises(BlockingIOError):
                        execute_job(args, run)
                audit.assert_not_called()
            self.assertEqual(json.loads((run/"receipt.json").read_text())["status"], "failed")


class WorkerCPUIsolationRuntimeTests(unittest.TestCase):
    def test_remote_full_suite_inherits_cpu_affinity_priority_memory_and_hidden_cuda(self):
        import resource
        import torch
        from ops.biohub import CPU_ENV, CPU_PROFILE, CPU_ADDRESS_SPACE_BYTES, verify_cpu_runtime
        self.assertEqual(os.environ.get("BIOHUB_RESOURCE_PROFILE"), CPU_PROFILE)
        self.assertTrue(all(os.environ.get(k) == v for k, v in CPU_ENV.items()))
        self.assertLessEqual(len(os.sched_getaffinity(0)), 2)
        self.assertGreaterEqual(os.nice(0), 10)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        self.assertTrue(0 < soft <= hard <= CPU_ADDRESS_SPACE_BYTES)
        self.assertEqual(verify_cpu_runtime(torch)["cuda_visible_devices"], 0)


if __name__ == "__main__":
    unittest.main()
