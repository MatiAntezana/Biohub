"""CPU-only tests for the Kaggle-guided autonomous variant policy."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))

from experimentlib import atomic_write_json, validate_kaggle_assets, validate_manifest  # noqa: E402
from kaggle_scores import append_score  # noqa: E402
from variant_planner import (  # noqa: E402
    _lane_counts,
    latest_proposals,
    maintain_autonomous_queue,
    materialize,
    plan,
    read_events,
    verify_ledger,
)
from variant_kaggle import prepare_one  # noqa: E402


class VariantPlannerTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        root = Path(temporary)
        for directory in (
            ".agents",
            ".state",
            "experiments",
            "infra/templates",
            "src/strategies",
            "src/models",
            "Librería-Registros/Experimentos",
            "experiments/EXP-017_Wide_UNet3D_90_10/kaggle",
        ):
            (root / directory).mkdir(parents=True, exist_ok=True)
        for relative in (
            ".agents/autonomous-exploration.json",
            ".agents/variant-recipes.json",
            "infra/templates/autonomous_variant_train.py",
            "src/strategies/autonomous_detection_training.py",
            "src/strategies/experiment_runtime.py",
            "src/dataset.py",
            "src/models/unet3d_anisotropic.py",
            "src/models/resnet_unet3d.py",
            "src/models/unet3d_groupnorm.py",
            "src/models/attention_unet3d.py",
            "src/models/convnext_unet3d.py",
            "experiments/EXP-017_Wide_UNet3D_90_10/kaggle/inference.py",
            "experiments/EXP-017_Wide_UNet3D_90_10/kaggle/build_nb.py",
        ):
            source = ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        atomic_write_json(
            root / ".agents/nodes.json",
            {
                "schema_version": 1,
                "nodes": {
                    name: {"host": "127.0.0.1", "conda_env": f"train-{name[-2:]}"}
                    for name in ("A1552", "A1556", "A1557", "A1560")
                },
            },
        )
        atomic_write_json(root / ".state/queue.json", {"schema_version": 1, "items": []})
        append_score(
            experiment_id="EXP-017",
            status="SCORED",
            source="test",
            submission_ref="17",
            public_score=0.683,
            root=root,
        )
        append_score(
            experiment_id="EXP-019",
            status="SCORED",
            source="test",
            submission_ref="19",
            public_score=0.643,
            root=root,
        )
        append_score(
            experiment_id="EXP-000",
            status="SCORED",
            source="test",
            submission_ref="0",
            public_score=0.99,
            root=root,
        )
        return root

    def test_full_batch_reserves_exploitation_and_both_exploration_lanes(self) -> None:
        policy = json.loads((ROOT / ".agents/autonomous-exploration.json").read_text())
        self.assertEqual(
            _lane_counts(5, policy),
            {"exploitation": 3, "adjacent_exploration": 1, "novel_exploration": 1},
        )

    def test_plan_is_deterministic_excludes_baseline_and_limits_parent_streak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            first = plan(5, seed=1234, root=root, persist=False)
            second = plan(5, seed=1234, root=root, persist=False)
            self.assertEqual(first, second)
            self.assertNotIn("EXP-000", {row["parent_experiment_id"] for row in first})
            parents = [row["parent_experiment_id"] for row in first]
            self.assertFalse(any(parents[index:index + 3] == [parents[index]] * 3 for index in range(len(parents) - 2)))
            self.assertEqual([row["lane"] for row in first].count("exploitation"), 3)

    def test_bootstrap_blocks_generation_until_exp031_and_exp030b_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            result = maintain_autonomous_queue(root=root, dry_run=True)
            self.assertEqual(result["blocked_by_bootstrap"], ["EXP-031", "EXP-030B"])
            self.assertEqual(read_events(root), [])

    def test_materialization_is_isolated_valid_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            proposal = plan(1, seed=55, root=root, persist=True)[0]
            result = materialize(proposal, root=root, auto_queue=True)
            directory = root / result["directory"]
            self.assertTrue((directory / "variant.json").is_file())
            self.assertTrue((directory / "train.py").is_file())
            self.assertTrue((directory / "kaggle/inference.py").is_file())
            self.assertIn("BIOHUB_TEST_ONLY_GUARD", (directory / "kaggle/inference.py").read_text())
            self.assertEqual(validate_manifest(proposal["experiment_id"], root), [])
            self.assertEqual(validate_kaggle_assets(proposal["experiment_id"], root), [])
            self.assertEqual(result["queue_status"], "READY")
            self.assertEqual(verify_ledger(root), [])
            self.assertEqual(latest_proposals(root)[proposal["proposal_id"]]["status"], "READY")

            checkpoint = directory / "checkpoints/run-test/training_checkpoint.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"dry-run-does-not-load")
            preview = prepare_one(proposal, root=root, dry_run=True)
            self.assertTrue(preview["would_upload_dataset"])
            self.assertTrue(preview["would_enable_deployment"])


if __name__ == "__main__":
    unittest.main()
