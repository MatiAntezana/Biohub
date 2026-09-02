"""Network-free tests for post-training automatic Kaggle publication."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

import automatic_mode  # noqa: E402
from experimentlib import SUBMISSION_COLUMNS, append_event, atomic_write_json  # noqa: E402


class MemoryStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def read(self) -> dict:
        return {"schema_version": 1, "jobs": self.jobs}

    def save_job(self, job: dict) -> None:
        self.jobs[job["job_id"]] = dict(job)


class FakePipeline:
    def __init__(self) -> None:
        self.store = MemoryStore()
        self.pushes: list[str] = []

    def push(self, experiment_id: str) -> dict:
        self.pushes.append(experiment_id)
        return {
            "job_id": f"job-{len(self.pushes)}",
            "experiment_id": experiment_id,
            "kernel_slug": f"owner/{experiment_id.lower()}",
            "package_sha256": "filled-by-controller",
            "state": "RUNNING",
            "created_at": "2026-08-29T12:00:00Z",
        }


class AutomaticKagglePublicationTests(unittest.TestCase):
    def _project(self, temporary: str, *, deployment_ready: bool = True) -> Path:
        root = Path(temporary)
        (root / ".agents").mkdir()
        (root / ".state").mkdir()
        (root / "experiments").mkdir()
        atomic_write_json(
            root / ".agents" / "kaggle-automation.json",
            {
                "schema_version": 1,
                "enabled_in_automatic_mode": True,
                "max_concurrent_kaggle_jobs": 2,
                "max_push_attempts_per_deployment": 3,
                "push_retry_backoff_seconds": 0,
            },
        )
        atomic_write_json(root / ".state" / "queue.json", {"schema_version": 1, "items": []})
        directory = root / "experiments" / "EXP-700_Test"
        (directory / "kaggle").mkdir(parents=True)
        (directory / "train.py").write_text("print('remote')\n", encoding="utf-8")
        (directory / "kaggle" / "inference.py").write_text("BIOHUB_TEST_ONLY_GUARD=True\n", encoding="utf-8")
        atomic_write_json(
            directory / "experiment.json",
            {
                "schema_version": 1,
                "experiment_id": "EXP-700",
                "entrypoint": "train.py",
                "arguments": [],
                "allowed_nodes": ["A"],
                "default_node": "A",
                "execution_policy": {"node_selection": "auto", "min_free_vram_mib": 0, "auto_launch": True, "max_parallel_runs": 1},
                "priority": 10,
                "dependencies": [],
                "data_policy": {},
                "artifact_contract": {"terminal_receipt": "x", "checkpoints": ["x"], "minimum_checkpoints": 1, "maximum_checkpoints": 1},
                "documentation": {},
                "kaggle": {
                    "enabled": True,
                    "deployment_ready": deployment_ready,
                    "auto_publish_after_train": True,
                    "auto_push": True,
                    "auto_submit": True,
                    "training_allowed": False,
                    "test_only": True,
                    "accelerator": "NvidiaTeslaT4",
                    "kernel_slug": "owner/exp-700",
                    "submission_columns": SUBMISSION_COLUMNS,
                },
            },
        )
        append_event(
            experiment_id="EXP-700",
            run_id="EXP-700_run-1",
            event_type="COMPLETED",
            status="SUCCEEDED",
            root=root,
            idempotency_key="run-1-success",
        )
        return root

    def test_successful_ready_run_is_pushed_once_without_manual_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            pipeline = FakePipeline()
            with mock.patch.object(automatic_mode, "KagglePipeline", return_value=pipeline):
                first = automatic_mode._push_new_successes(root, {"events": []})
                second = automatic_mode._push_new_successes(root, {"events": []})
            self.assertEqual(pipeline.pushes, ["EXP-700"])
            self.assertEqual(first["pushed"][0]["run_id"], "EXP-700_run-1")
            self.assertEqual(second["waiting"][0]["reason"], "kernel version already active")
            job = next(iter(pipeline.store.jobs.values()))
            self.assertEqual(job["source_run_id"], "EXP-700_run-1")
            self.assertIn("EXP-700_run-1", job["deployment_key"])

    def test_new_successful_run_gets_new_version_after_previous_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            pipeline = FakePipeline()
            with mock.patch.object(automatic_mode, "KagglePipeline", return_value=pipeline):
                automatic_mode._push_new_successes(root, {"events": []})
                pipeline.store.jobs["job-1"]["state"] = "SCORED"
                append_event(
                    experiment_id="EXP-700",
                    run_id="EXP-700_run-2",
                    event_type="COMPLETED",
                    status="SUCCEEDED",
                    root=root,
                    idempotency_key="run-2-success",
                )
                result = automatic_mode._push_new_successes(root, {"events": []})
            self.assertEqual(pipeline.pushes, ["EXP-700", "EXP-700"])
            self.assertEqual(result["pushed"][0]["run_id"], "EXP-700_run-2")

    def test_deployment_not_ready_waits_without_network_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary, deployment_ready=False)
            pipeline = FakePipeline()
            with mock.patch.object(automatic_mode, "KagglePipeline", return_value=pipeline):
                result = automatic_mode._push_new_successes(root, {"events": []})
            self.assertEqual(pipeline.pushes, [])
            self.assertEqual(result["waiting"][0]["reason"], "deployment_ready=false")


if __name__ == "__main__":
    unittest.main()
