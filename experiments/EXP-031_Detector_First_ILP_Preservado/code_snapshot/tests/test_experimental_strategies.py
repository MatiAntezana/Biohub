"""Small CPU-only regression tests for EXP-030/031 strategy primitives."""

from __future__ import annotations

import unittest

import torch

from src.strategies.detector_precision import precision_focal_detection_loss
from src.strategies.ensemble_temporal import (
    align_secondary_logits,
    d4_xy_transforms,
    retention_guard,
    weighted_harmonic_forward_reverse,
)
from src.strategies.ilp_preserving_repair import (
    RepairConfig,
    TrackingGraph,
    add_conservative_divisions,
    audit_graph,
    preserve_confident_ilp_edges,
    repair_unmatched_motion,
)


class ExperimentalStrategiesTests(unittest.TestCase):
    def test_secondary_alignment_matches_primary_statistics(self) -> None:
        primary = torch.tensor([[[-2.0, 0.0], [2.0, 4.0]]])
        secondary = torch.tensor([[[10.0, 11.0], [12.0, 13.0]]])
        aligned = align_secondary_logits(primary, secondary)
        self.assertAlmostEqual(aligned.mean().item(), primary.mean().item(), places=6)
        self.assertAlmostEqual(
            aligned.std(unbiased=False).item(),
            primary.std(unbiased=False).item(),
            places=6,
        )

    def test_retention_guard_falls_back_per_frame(self) -> None:
        primary = torch.ones(4)
        fused = torch.zeros(4)
        chosen, report = retention_guard(
            primary,
            fused,
            candidate_counter=lambda value: int((value > 0.5).sum().item()),
            minimum=0.9,
        )
        self.assertIs(chosen, primary)
        self.assertTrue(report["fallback"])
        self.assertEqual(report["retention"], 0.0)

    def test_all_d4_transforms_are_invertible(self) -> None:
        image = torch.arange(16).reshape(1, 1, 4, 4)
        names = []
        for name, transform, inverse in d4_xy_transforms():
            names.append(name)
            self.assertTrue(torch.equal(inverse(transform(image)), image))
        self.assertEqual(len(set(names)), 8)

    def test_weighted_harmonic_support_is_bounded(self) -> None:
        forward = torch.tensor([[0.8, 0.4]])
        reverse = torch.tensor([[0.5, 0.9]])
        fused = weighted_harmonic_forward_reverse(forward, reverse, reverse_weight=0.3)
        self.assertTrue(torch.all(fused >= torch.minimum(forward, reverse)))
        self.assertTrue(torch.all(fused <= torch.maximum(forward, reverse)))

    def test_precision_loss_has_finite_gradient(self) -> None:
        logits = torch.zeros((1, 1, 5, 9, 9), requires_grad=True)
        coords = torch.tensor([[[2.0, 4.0, 4.0]]])
        mask = torch.tensor([[True]])
        loss = precision_focal_detection_loss(
            logits,
            coords,
            mask,
            neg_weight=0.25,
            hard_negative_ratio=4,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_motion_repair_preserves_ilp_and_only_fills_unmatched(self) -> None:
        nodes = {
            1: (0, 0.0, 0.0, 0.0),
            2: (1, 0.0, 1.0, 0.0),
            3: (0, 0.0, 10.0, 0.0),
            4: (1, 0.0, 11.0, 0.0),
        }
        initial = TrackingGraph(nodes, {(1, 2): 0.95, (1, 4): 0.2})
        config = RepairConfig(scale_um=(1.0, 1.0, 1.0), min_ilp_confidence=0.48)
        preserved = preserve_confident_ilp_edges(initial, config)
        repaired, report = repair_unmatched_motion(
            preserved,
            learned_probabilities={(3, 4): 0.9},
            config=config,
        )
        self.assertIn((1, 2), repaired.edges)
        self.assertIn((3, 4), repaired.edges)
        self.assertNotIn((1, 4), repaired.edges)
        self.assertEqual(report["accepted_tight"], 1)
        self.assertEqual(audit_graph(repaired)["edges"], 2)

    def test_audit_rejects_nonconsecutive_edges(self) -> None:
        graph = TrackingGraph(
            {1: (0, 0.0, 0.0, 0.0), 2: (2, 0.0, 0.0, 0.0)},
            {(1, 2): 1.0},
        )
        with self.assertRaisesRegex(ValueError, "invalid tracking graph"):
            audit_graph(graph)

    def test_division_requires_continuing_divergent_branches(self) -> None:
        graph = TrackingGraph(
            {
                0: (0, 0.0, 0.0, 0.0),
                1: (1, 0.0, 0.0, 0.0),
                2: (2, 0.0, 1.0, 0.0),
                3: (2, 0.0, -1.0, 0.0),
                4: (3, 0.0, 4.0, 0.0),
                5: (3, 0.0, -4.0, 0.0),
            },
            {(0, 1): 0.9, (1, 2): 0.9, (2, 4): 0.9, (3, 5): 0.9},
        )
        config = RepairConfig(scale_um=(1.0, 1.0, 1.0))
        repaired, report = add_conservative_divisions(
            graph,
            candidate_probabilities={(1, 3): 0.85},
            config=config,
        )
        self.assertIn((1, 3), repaired.edges)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(audit_graph(repaired)["outdegree_violations"], 0)


if __name__ == "__main__":
    unittest.main()
