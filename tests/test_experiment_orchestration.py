"""CPU-only tests for the deterministic Biohub experiment controller."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

import experiment_worker  # noqa: E402
import experimentctl  # noqa: E402
import experiment_scheduler  # noqa: E402
import agent_commands  # noqa: E402
import automatic_mode  # noqa: E402
from gpu_selector import parse_nvidia_smi, select_gpu  # noqa: E402
from remote_gpu_monitor import _transitions as gpu_transitions, infer_experiment_id, parse_remote_payload, scan_remote_gpus  # noqa: E402
from kaggle_pipeline import inspect_runtime_log, static_preflight, validate_submission_csv  # noqa: E402
from promote_best import ranked_candidates  # noqa: E402
from aienv import ENVIRONMENT_NAME, in_aienv  # noqa: E402
from experimentlib import (  # noqa: E402
    SUBMISSION_COLUMNS,
    append_event,
    atomic_write_json,
    experiment_directory,
    latest_status_for_experiment,
    read_events,
    refresh_queue,
    render_views,
    validate_document_link,
    validate_kaggle_assets,
    validate_manifest,
    verify_ledger,
)


def _append_many(root: str, worker: int) -> None:
    path = Path(root)
    for index in range(8):
        append_event(
            experiment_id="EXP-999",
            run_id=f"worker-{worker}",
            event_type="TEST",
            status="RUNNING",
            root=path,
            idempotency_key=f"{worker}:{index}",
        )


def _manifest(experiment_id: str, *, dependency: str | None = None, legacy: bool = False) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "entrypoint": "train.py",
        "arguments": [],
        "allowed_nodes": ["A1556"],
        "default_node": "A1556",
        "execution_policy": {
            "node_selection": "max_free_vram",
            "min_free_vram_mib": 0,
            "auto_launch": False,
            "max_parallel_runs": 1,
        },
        "dependencies": [dependency] if dependency else [],
        "data_policy": {"validation": "none"},
        "artifact_contract": {
            "terminal_receipt": "artifacts/{run_id}/TRAINING_COMPLETE.json",
            "checkpoints": ["checkpoints/{run_id}/model/training_checkpoint.pth"],
            "minimum_checkpoints": 1,
            "maximum_checkpoints": 1,
        },
        "documentation": {"narrative": "docs/test.md"},
        "kaggle": {
            "enabled": True,
            "deployment_ready": False,
            "training_allowed": False,
            "test_only": True,
            "accelerator": "NvidiaTeslaT4",
            "submission_columns": SUBMISSION_COLUMNS,
        },
        "note": "experiment_artifacts" if legacy else "isolated",
    }


class OrchestrationTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / ".agents").mkdir()
        (root / ".state").mkdir()
        (root / "experiments").mkdir()
        atomic_write_json(
            root / ".agents" / "nodes.json",
            {"schema_version": 1, "nodes": {"A1556": {"host": "127.0.0.1"}}},
        )
        atomic_write_json(root / ".state" / "queue.json", {"schema_version": 1, "items": []})
        return root

    def _experiment(self, root: Path, experiment_id: str, manifest: dict) -> Path:
        directory = root / "experiments" / f"{experiment_id}_Test"
        for name in ("artifacts", "logs", "checkpoints", "code_snapshot", "validation", "kaggle"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        (directory / "train.py").write_text("print('test')\n", encoding="utf-8")
        atomic_write_json(directory / "experiment.json", manifest)
        return directory

    def test_registry_is_hash_chained_concurrent_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            context = multiprocessing.get_context("fork")
            processes = [context.Process(target=_append_many, args=(str(root), worker)) for worker in range(4)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(len(read_events(root)), 32)
            self.assertEqual(verify_ledger(root), [])
            append_event(
                experiment_id="EXP-999",
                run_id="worker-0",
                event_type="TEST",
                status="RUNNING",
                root=root,
                idempotency_key="0:0",
            )
            self.assertEqual(len(read_events(root)), 32)

    def test_registry_rejects_invalid_state_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            append_event(
                experiment_id="EXP-998",
                run_id="transition-run",
                event_type="STARTED",
                status="RUNNING",
                root=root,
                idempotency_key="started",
            )
            with self.assertRaisesRegex(Exception, "transición inválida"):
                append_event(
                    experiment_id="EXP-998",
                    run_id="transition-run",
                    event_type="CLOSED",
                    status="CLOSED",
                    root=root,
                    idempotency_key="closed-too-early",
                )
            self.assertEqual(verify_ledger(root), [])
            self.assertEqual(len(read_events(root)), 1)

    def test_dependency_only_becomes_ready_and_never_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            self._experiment(root, "EXP-100", _manifest("EXP-100"))
            self._experiment(root, "EXP-101", _manifest("EXP-101", dependency="EXP-100"))
            atomic_write_json(
                root / ".state" / "queue.json",
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "experiment_id": "EXP-101",
                            "status": "QUEUED",
                            "dependencies": ["EXP-100"],
                            "priority": 1,
                            "requested_node": "A1556",
                        }
                    ],
                },
            )
            append_event(
                experiment_id="EXP-100",
                run_id="parent-run",
                event_type="COMPLETED",
                status="SUCCEEDED",
                root=root,
                idempotency_key="parent-complete",
            )
            queue = refresh_queue(root)
            self.assertEqual(queue["items"][0]["status"], "READY")
            self.assertNotEqual(queue["items"][0]["status"], "RUNNING")

    def test_scheduler_only_considers_explicit_auto_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            manual = _manifest("EXP-110")
            automatic = _manifest("EXP-111")
            automatic["execution_policy"]["auto_launch"] = True
            self._experiment(root, "EXP-110", manual)
            self._experiment(root, "EXP-111", automatic)
            atomic_write_json(
                root / ".state" / "queue.json",
                {
                    "schema_version": 1,
                    "items": [
                        {"experiment_id": "EXP-110", "status": "READY", "dependencies": [], "priority": 1, "requested_node": "max_free_vram", "auto_launch": False},
                        {"experiment_id": "EXP-111", "status": "READY", "dependencies": [], "priority": 2, "requested_node": "max_free_vram", "auto_launch": True},
                    ],
                },
            )
            self.assertEqual([item["experiment_id"] for item in experiment_scheduler.eligible_items(root)], ["EXP-111"])

    def test_exact_operator_phrases_are_deterministically_mapped(self) -> None:
        self.assertEqual(agent_commands.COMMANDS["cortar train remoto"], ["experimentctl.py", "stop", "--all"])
        self.assertEqual(agent_commands.COMMANDS["subir mejores versiones a kaggle"], ["promote_best.py", "--limit", "3"])
        self.assertEqual(agent_commands.COMMANDS["modo automatico"], ["automatic_mode.py", "supervise"])
        self.assertEqual(agent_commands.COMMANDS["cortar automatico"], ["automatic_mode.py", "stop"])

    def test_attached_monitor_reports_compact_training_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            directory = self._experiment(root, "EXP-120", _manifest("EXP-120"))
            run_id = "EXP-120_run"
            run_dir = directory / "artifacts" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                root / ".state" / "queue.json",
                {
                    "schema_version": 1,
                    "items": [{"experiment_id": "EXP-120", "status": "RUNNING", "run_id": run_id, "node": "A1556"}],
                },
            )
            atomic_write_json(root / ".state" / "automatic-mode.json", {"status": "RUNNING", "pid": 123, "events": []})
            atomic_write_json(
                run_dir / "metrics.json",
                [{"epoch": 4, "edge_loss": 0.2, "det_loss": 0.1, "large_payload": "ignored"}],
            )
            snapshot = automatic_mode.monitoring_snapshot(root)
            self.assertEqual(snapshot["runs"][0]["progress"]["main"], {"epoch": 4, "edge_loss": 0.2, "det_loss": 0.1})
        compile(experimentctl._remote_stop_code(Path("experiments/EXP-999/artifacts/run/RECEIPT.json")), "<remote-stop>", "exec")

    def test_manifest_rejects_legacy_shared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            self._experiment(root, "EXP-102", _manifest("EXP-102", legacy=True))
            errors = validate_manifest("EXP-102", root)
            self.assertTrue(any("ruta legacy" in error for error in errors))

    def test_views_are_generated_from_structured_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            append_event(
                experiment_id="EXP-999",
                run_id="test-run",
                event_type="FAILED",
                status="FAILED",
                root=root,
                idempotency_key="failed",
            )
            render_views(root)
            self.assertIn("test-run", (root / ".state" / "STATE.md").read_text())
            self.assertIn("test-run", (root / "experiments" / "RESULTS.md").read_text())

    def test_document_link_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            document = root / "docs" / "result.md"
            document.parent.mkdir()
            document.write_text("conclusión", encoding="utf-8")
            self.assertEqual(validate_document_link("docs/result.md", root), "docs/result.md")
            with self.assertRaisesRegex(Exception, "inexistente"):
                validate_document_link("docs/missing.md", root)

    def test_worker_success_and_failure_emit_terminal_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            directory = root / "experiments" / "EXP-103_Test"
            for name in ("artifacts/run-ok", "artifacts/run-fail", "logs", "checkpoints/run-ok/model"):
                (directory / name).mkdir(parents=True, exist_ok=True)
            (root / "data" / "train").mkdir(parents=True)
            success_script = directory / "success.py"
            success_script.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser(); p.add_argument('--run-dir'); a,_=p.parse_known_args()\n"
                "r=Path(a.run_dir); e=r.parents[1]; c=e/'checkpoints'/r.name/'model'; c.mkdir(parents=True,exist_ok=True)\n"
                "(c/'training_checkpoint.pth').write_bytes(b'checkpoint')\n"
                "(r/'TRAINING_COMPLETE.json').write_text(json.dumps({'metrics':{'loss':1.0}}))\n",
                encoding="utf-8",
            )
            fail_script = directory / "fail.py"
            fail_script.write_text("raise SystemExit(3)\n", encoding="utf-8")

            def spec(run_id: str, entrypoint: str) -> Path:
                path = directory / "artifacts" / run_id / "run_spec.json"
                atomic_write_json(
                    path,
                    {
                        "run_id": run_id,
                        "experiment_id": "EXP-103",
                        "remote_root": str(root),
                        "experiment_directory": "experiments/EXP-103_Test",
                        "entrypoint": entrypoint,
                        "arguments": [],
                        "run_directory": f"experiments/EXP-103_Test/artifacts/{run_id}",
                        "artifact_contract": {
                            "terminal_receipt": "artifacts/{run_id}/TRAINING_COMPLETE.json",
                            "checkpoints": ["checkpoints/{run_id}/model/training_checkpoint.pth"],
                            "minimum_checkpoints": 1,
                            "maximum_checkpoints": 1,
                        },
                        "data_policy": {"validation": "none"},
                        "config_sha256": "abc",
                        "heartbeat_seconds": 1,
                    },
                )
                return path

            environment = {"conda_env": None, "gpu": ["mock"]}
            with mock.patch.object(experiment_worker, "_environment", return_value=environment):
                succeeded = experiment_worker.run(spec("run-ok", "success.py"))
            self.assertEqual(succeeded["status"], "SUCCEEDED", succeeded.get("error"))
            self.assertEqual(succeeded["metrics"]["loss"], 1.0)

            with mock.patch.object(experiment_worker, "_environment", return_value=environment):
                failed = experiment_worker.run(spec("run-fail", "fail.py"))
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["exit_code"], 3)

    def test_active_manifests_match_kaggle_contract(self) -> None:
        for experiment_id in ("EXP-030", "EXP-030B", "EXP-031"):
            self.assertEqual(validate_manifest(experiment_id, ROOT), [])
            self.assertEqual(validate_kaggle_assets(experiment_id, ROOT), [])
        self.assertEqual(latest_status_for_experiment("EXP-404", ROOT), None)

    def test_suite_runs_in_canonical_local_environment(self) -> None:
        self.assertEqual(ENVIRONMENT_NAME, "AIEnv")
        self.assertTrue(in_aienv())

    def test_remote_pool_declares_verified_environments(self) -> None:
        nodes = json.loads((ROOT / ".agents" / "nodes.json").read_text(encoding="utf-8"))["nodes"]
        self.assertEqual(nodes["A1552"]["conda_env"], "train-52")
        self.assertEqual(nodes["A1556"]["conda_env"], "train-56")
        self.assertEqual(nodes["A1557"]["conda_env"], "train-57")
        self.assertEqual(nodes["A1560"]["conda_env"], "train-60")
        self.assertEqual(nodes["A1560"]["status"], "OFFLINE")
        with self.assertRaisesRegex(Exception, "no está habilitada"):
            experimentctl._probe_node("A1560", nodes["A1560"])

    def test_gpu_selector_chooses_most_free_vram_and_excludes_offline(self) -> None:
        nodes = {
            "A": {"status": "ONLINE", "training_ready": True, "host": "a", "port": 1, "user": "u"},
            "B": {"status": "ONLINE", "training_ready": True, "host": "b", "port": 1, "user": "u"},
            "C": {"status": "OFFLINE", "training_ready": False, "host": "c", "port": 1, "user": "u"},
        }

        def runner(command, **_kwargs):
            host = next(part for part in command if "@" in part)
            output = "0, 12000, 24576, 10\n1, 22000, 24576, 5\n" if host.endswith("@a") else "0, 23000, 24576, 40\n"
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

        selected, probes = select_gpu(nodes, ["A", "B", "C"], min_free_vram_mib=18000, runner=runner)
        self.assertEqual((selected.node, selected.gpu_index, selected.free_mib), ("B", 0, 23000))
        self.assertEqual(probes["C"]["error"], "status=OFFLINE")
        self.assertEqual(parse_nvidia_smi("0, 123, 456, 7\n", "A")[0].free_mib, 123)

    def test_remote_gpu_monitor_identifies_experiments_and_transitions(self) -> None:
        payload = {
            "gpu_rc": 0,
            "gpu_out": "0, GPU-abc, NVIDIA RTX 4090, 24564, 18000, 6564, 92\n",
            "app_rc": 0,
            "app_out": "GPU-abc, 4242, python, 17600 MiB\n",
            "commands": {"4242": "python experiments/EXP-031_Detector/train.py --compact"},
        }
        parsed = parse_remote_payload("A1552", payload)
        activity = parsed["activities"][0]
        self.assertEqual(activity["experiment_id"], "EXP-031")
        self.assertEqual(activity["gpu_index"], 0)
        self.assertEqual(infer_experiment_id("python exp030b/train.py"), "EXP-030B")
        transitions = gpu_transitions({"activities": []}, {"activities": [activity]})
        self.assertEqual(transitions[0]["type"], "GPU_PROCESS_STARTED")

        nodes = {
            "A1552": {"status": "ONLINE", "training_ready": True, "host": "a", "port": 1, "user": "u"},
            "A1560": {"status": "OFFLINE", "training_ready": False, "host": "b", "port": 1, "user": "u"},
        }

        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        scan = scan_remote_gpus(nodes, runner=runner)
        self.assertEqual(set(scan["nodes"]), {"A1552"})
        self.assertEqual(scan["activities"][0]["experiment_id"], "EXP-031")

    def test_kaggle_log_guard_and_submission_schema(self) -> None:
        safe = inspect_runtime_log(
            'BIOHUB_DATASET_SCOPE={"scope":"test","paths":["/kaggle/input/c/test/a.zarr"]}\n'
            "BIOHUB_DATASET_PATH=/kaggle/input/c/test/a.zarr\n",
            require_scope=True,
        )
        self.assertEqual(safe["violations"], [])
        unsafe = inspect_runtime_log(
            'BIOHUB_DATASET_SCOPE={"scope":"train","paths":["/kaggle/input/c/train/a.zarr"]}\n',
            require_scope=True,
        )
        self.assertTrue(unsafe["violations"])
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "submission.csv"
            csv_path.write_text(
                ",".join(SUBMISSION_COLUMNS) + "\n0,test_a,node,0,0,1,2,3,-1,-1\n",
                encoding="utf-8",
            )
            summary = validate_submission_csv(csv_path)
            self.assertEqual(summary["nodes"], 1)

    def test_exp030_kaggle_package_is_strict_test_only(self) -> None:
        self.assertEqual(static_preflight("EXP-030", ROOT), [])

    def test_current_top_three_are_ranked_from_global_evaluator_and_test_only(self) -> None:
        candidates, _ = ranked_candidates(ROOT)
        self.assertEqual([row["experiment_id"] for row in candidates[:3]], ["EXP-015", "EXP-017", "EXP-022"])
        for experiment_id in ("EXP-015", "EXP-017", "EXP-022"):
            self.assertEqual(static_preflight(experiment_id, ROOT), [])

    def test_normalized_trainer_persists_resume_state(self) -> None:
        trainer = (ROOT / "official_starter_kit" / "scripts" / "train_unet_transformer.py").read_text()
        for key in (
            '"model_state_dict"',
            '"optimizer_state_dict"',
            '"scaler_state_dict"',
            '"next_epoch"',
            '"rng_state"',
        ):
            self.assertIn(key, trainer)
        for experiment_id in ("EXP-030", "EXP-030B", "EXP-031"):
            launcher = experiment_directory(experiment_id, ROOT) / "train.py"
            source = launcher.read_text(encoding="utf-8")
            self.assertIn("checkpoint_bundle=True", source)
            self.assertIn("progress=not args.compact", source)


if __name__ == "__main__":
    unittest.main()
