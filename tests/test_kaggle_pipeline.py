"""Network-free integration tests for the Kaggle policy pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

from experimentlib import SUBMISSION_COLUMNS, atomic_write_json  # noqa: E402
from kaggle_pipeline import KagglePipeline  # noqa: E402


class FakeKaggle:
    def __init__(self, *, unsafe_log: bool = False, kernel_status: str = "COMPLETE", error_log: str = "") -> None:
        self.commands: list[list[str]] = []
        self.message = ""
        self.unsafe_log = unsafe_log
        self.kernel_status = kernel_status
        self.error_log = error_log

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        stdout = ""
        if "push" in command:
            stdout = "Kernel version 12 successfully pushed. Please check progress at https://kaggle.test"
        elif "status" in command:
            stdout = f'{command[-1]} has status "{self.kernel_status}"'
        elif "logs" in command:
            scope = "train" if self.unsafe_log else "test"
            stdout = (
                f'BIOHUB_DATASET_SCOPE={{"scope":"{scope}","paths":["/kaggle/input/c/{scope}/a.zarr"]}}\n'
                f"BIOHUB_DATASET_PATH=/kaggle/input/c/{scope}/a.zarr\n"
                f"{self.error_log}\n"
            )
        elif "output" in command:
            output_dir = Path(command[command.index("-p") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "submission.csv").write_text(
                ",".join(SUBMISSION_COLUMNS) + "\n0,test_a,node,0,0,1,2,3,-1,-1\n",
                encoding="utf-8",
            )
        elif "submit" in command:
            self.message = command[command.index("-m") + 1]
            stdout = "Successfully submitted to competition"
        elif "submissions" in command:
            stdout = (
                "fileName,date,description,status,publicScore,privateScore\n"
                f"submission.csv,now,{self.message},complete,0.123,\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


class KagglePipelineTests(unittest.TestCase):
    def _project(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / ".state").mkdir()
        (root / ".agents").mkdir()
        (root / "experiments").mkdir()
        directory = root / "experiments" / "EXP-200_Test"
        for name in ("artifacts", "logs", "checkpoints", "code_snapshot", "validation", "kaggle"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        (directory / "train.py").write_text("print('remote only')\n", encoding="utf-8")
        atomic_write_json(
            root / ".agents" / "nodes.json",
            {"schema_version": 1, "nodes": {"A": {"host": "none"}}},
        )
        atomic_write_json(root / ".state" / "queue.json", {"schema_version": 1, "items": []})
        atomic_write_json(
            directory / "experiment.json",
            {
                "schema_version": 1,
                "experiment_id": "EXP-200",
                "entrypoint": "train.py",
                "arguments": [],
                "allowed_nodes": ["A"],
                "default_node": "A",
                "execution_policy": {"node_selection": "max_free_vram", "min_free_vram_mib": 0, "auto_launch": False, "max_parallel_runs": 1},
                "dependencies": [],
                "data_policy": {},
                "artifact_contract": {"terminal_receipt": "x", "checkpoints": ["x"], "minimum_checkpoints": 1, "maximum_checkpoints": 1},
                "documentation": {},
                "kaggle": {
                    "enabled": True,
                    "deployment_ready": True,
                    "training_allowed": False,
                    "test_only": True,
                    "accelerator": "NvidiaTeslaT4",
                    "competition": "biohub-cell-tracking-during-development",
                    "kernel_slug": "owner/kernel",
                    "output_file": "submission.csv",
                    "auto_submit": True,
                    "max_retries": 2,
                    "submission_columns": SUBMISSION_COLUMNS,
                },
            },
        )
        kaggle = directory / "kaggle"
        (kaggle / "inference.py").write_text(
            'BIOHUB_TEST_ONLY_GUARD=True\nprint("BIOHUB_DATASET_SCOPE=")\nprint("BIOHUB_DATASET_PATH=")\n'
            + json.dumps(SUBMISSION_COLUMNS)
            + "\n",
            encoding="utf-8",
        )
        (kaggle / "build_nb.py").write_text("print('built')\n", encoding="utf-8")
        atomic_write_json(
            kaggle / "kernel-metadata.json",
            {"id": "owner/kernel", "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4"},
        )
        atomic_write_json(
            kaggle / "submission_notebook.ipynb",
            {"metadata": {"kernelspec": {"name": "python3"}}, "cells": [], "nbformat": 4, "nbformat_minor": 4},
        )
        return root

    def test_complete_test_run_is_validated_submitted_and_scored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            fake = FakeKaggle()
            pipeline = KagglePipeline(root, runner=fake)
            job = pipeline.push("EXP-200")
            self.assertEqual(job["kernel_version"], 12)
            job = pipeline.watch_once(job["job_id"])
            self.assertEqual(job["state"], "SUBMITTED")
            job = pipeline.watch_once(job["job_id"])
            self.assertEqual(job["state"], "SCORED")
            self.assertEqual(job["public_score"], 0.123)
            leaderboard = (root / "experiments" / "KAGGLE_LEADERBOARD.md").read_text(encoding="utf-8")
            self.assertIn("EXP-200", leaderboard)
            self.assertIn("0.1230", leaderboard)
            self.assertTrue(any("submit" in command for command in fake.commands))

    def test_train_scope_blocks_output_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            fake = FakeKaggle(unsafe_log=True)
            pipeline = KagglePipeline(root, runner=fake)
            job = pipeline.push("EXP-200")
            job = pipeline.watch_once(job["job_id"])
            self.assertEqual(job["state"], "POLICY_BLOCKED")
            self.assertFalse(any("submit" in command for command in fake.commands))
            self.assertFalse(any("output" in command for command in fake.commands))

    def test_transient_error_repushes_but_deterministic_error_waits_for_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            transient = FakeKaggle(kernel_status="ERROR", error_log="503 Service unavailable; worker lost")
            pipeline = KagglePipeline(root, runner=transient)
            job = pipeline.push("EXP-200")
            job = pipeline.watch_once(job["job_id"])
            self.assertEqual(job["state"], "RUNNING")
            self.assertEqual(job["retries"], 1)
            self.assertEqual(sum("push" in command for command in transient.commands), 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = self._project(temporary)
            deterministic = FakeKaggle(kernel_status="ERROR", error_log="Traceback: CUDA out of memory")
            pipeline = KagglePipeline(root, runner=deterministic)
            job = pipeline.push("EXP-200")
            job = pipeline.watch_once(job["job_id"])
            self.assertEqual(job["state"], "NEEDS_REPAIR")
            self.assertEqual(sum("push" in command for command in deterministic.commands), 1)


if __name__ == "__main__":
    unittest.main()
