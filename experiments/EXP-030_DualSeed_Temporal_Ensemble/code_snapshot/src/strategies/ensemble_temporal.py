"""Calibrated dual-seed and eight-view TTA primitives for EXP-030."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class EnsembleConfig:
    """Frozen inference parameters taken from the independent strategy."""

    detection_secondary_weight: float = 0.475
    edge_secondary_weight: float = 0.15
    retention_minimum: float = 0.90
    primary_margin_limit: float = 0.35
    reverse_weight: float = 0.30
    std_ratio_minimum: float = 0.5
    std_ratio_maximum: float = 2.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def align_secondary_logits(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    *,
    ratio_minimum: float = 0.5,
    ratio_maximum: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Align secondary mean/std to the primary without changing shape."""

    if primary.shape != secondary.shape:
        raise ValueError(f"logit shapes differ: {primary.shape} != {secondary.shape}")
    dims = tuple(range(1, primary.ndim))
    p_mean = primary.mean(dim=dims, keepdim=True)
    s_mean = secondary.mean(dim=dims, keepdim=True)
    p_std = primary.std(dim=dims, keepdim=True, unbiased=False)
    s_std = secondary.std(dim=dims, keepdim=True, unbiased=False)
    ratio = (p_std / s_std.clamp_min(eps)).clamp(ratio_minimum, ratio_maximum)
    return (secondary - s_mean) * ratio + p_mean


def fuse_detection_logits(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    config: EnsembleConfig,
) -> torch.Tensor:
    """Fuse two calibrated detection-logit tensors."""

    aligned = align_secondary_logits(
        primary,
        secondary,
        ratio_minimum=config.std_ratio_minimum,
        ratio_maximum=config.std_ratio_maximum,
    )
    weight = config.detection_secondary_weight
    return (1.0 - weight) * primary + weight * aligned


def retention_guard(
    primary: torch.Tensor,
    fused: torch.Tensor,
    candidate_counter: Callable[[torch.Tensor], int],
    minimum: float = 0.90,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Fallback to primary logits when fusion loses too many candidates."""

    primary_count = int(candidate_counter(primary))
    fused_count = int(candidate_counter(fused))
    retention = fused_count / max(primary_count, 1)
    fallback = primary_count > 0 and retention < minimum
    chosen = primary if fallback else fused
    return chosen, {
        "primary_candidates": primary_count,
        "fused_candidates": fused_count,
        "retention": retention,
        "fallback": fallback,
    }


def _parent_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Normalize over candidate parents; one child cannot have two parents."""

    return torch.softmax(logits, dim=-2)


def fuse_edges_low_margin_consensus(
    primary_logits: torch.Tensor,
    secondary_logits: torch.Tensor,
    config: EnsembleConfig,
) -> torch.Tensor:
    """Use the second seed only for ambiguous parents on which both agree."""

    if primary_logits.shape != secondary_logits.shape:
        raise ValueError("primary and secondary edge tensors must have equal shapes")
    if primary_logits.shape[-2] < 1:
        return primary_logits

    aligned_secondary = align_secondary_logits(
        primary_logits,
        secondary_logits,
        ratio_minimum=config.std_ratio_minimum,
        ratio_maximum=config.std_ratio_maximum,
    )
    primary_prob = _parent_probabilities(primary_logits)
    secondary_prob = _parent_probabilities(aligned_secondary)

    k = min(2, primary_prob.shape[-2])
    top_values, top_indices = torch.topk(primary_prob, k=k, dim=-2)
    if k == 1:
        margin = top_values[..., 0, :]
    else:
        margin = top_values[..., 0, :] - top_values[..., 1, :]
    primary_parent = top_indices[..., 0, :]
    secondary_parent = secondary_prob.argmax(dim=-2)
    consensus = primary_parent.eq(secondary_parent)

    ambiguity = ((config.primary_margin_limit - margin) / config.primary_margin_limit).clamp(0.0, 1.0)
    local_weight = config.edge_secondary_weight * ambiguity * consensus.to(primary_logits.dtype)
    local_weight = local_weight.unsqueeze(-2)
    return (1.0 - local_weight) * primary_logits + local_weight * aligned_secondary


def weighted_harmonic_forward_reverse(
    forward_probabilities: torch.Tensor,
    reverse_probabilities_aligned: torch.Tensor,
    reverse_weight: float = 0.30,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted harmonic support in the forward candidate space."""

    if forward_probabilities.shape != reverse_probabilities_aligned.shape:
        raise ValueError("forward/reverse probability tensors must have equal shapes")
    if not 0.0 <= reverse_weight <= 1.0:
        raise ValueError("reverse_weight must be within [0, 1]")
    forward_weight = 1.0 - reverse_weight
    forward = forward_probabilities.clamp_min(eps)
    reverse = reverse_probabilities_aligned.clamp_min(eps)
    return 1.0 / (forward_weight / forward + reverse_weight / reverse)


def d4_xy_transforms() -> tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]], ...]:
    """Return the eight invertible symmetries of the final XY plane."""

    identity = lambda value: value
    flip_x = lambda value: value.flip(-1)
    flip_y = lambda value: value.flip(-2)
    flip_xy = lambda value: value.flip((-2, -1))
    rot90 = lambda value: torch.rot90(value, 1, (-2, -1))
    rot270 = lambda value: torch.rot90(value, 3, (-2, -1))
    transpose = lambda value: value.transpose(-2, -1)
    anti_transpose = lambda value: torch.rot90(value.transpose(-2, -1), 2, (-2, -1))
    return (
        ("identity", identity, identity),
        ("flip_x", flip_x, flip_x),
        ("flip_y", flip_y, flip_y),
        ("flip_xy", flip_xy, flip_xy),
        ("rot90", rot90, rot270),
        ("rot270", rot270, rot90),
        ("transpose", transpose, transpose),
        ("anti_transpose", anti_transpose, anti_transpose),
    )


@torch.no_grad()
def eight_view_detection_logits(model, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Average detection logits across D4 views and retain identity features."""

    if images.shape[-1] != images.shape[-2]:
        raise ValueError("eight-view XY TTA requires equal Y/X dimensions")
    identity_features = None
    accumulated: list[torch.Tensor] | None = None
    for name, transform, inverse in d4_xy_transforms():
        features, detection = model.encode(transform(images))
        if name == "identity":
            identity_features = features
        restored = [inverse(frame_logits) for frame_logits in detection]
        if accumulated is None:
            accumulated = restored
        else:
            accumulated = [left + right for left, right in zip(accumulated, restored, strict=True)]
    assert identity_features is not None and accumulated is not None
    return identity_features, [value / 8.0 for value in accumulated]
