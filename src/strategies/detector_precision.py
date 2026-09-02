"""Precision-oriented detector loss for the audit-driven EXP-031 branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _point_target(
    logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rasterize annotated centers into a binary tensor matching ``logits``."""

    batch = logits.shape[0]
    spatial = logits.shape[2:]
    target = torch.zeros_like(logits[:, 0])
    for index in range(batch):
        count = int(mask[index].sum().item())
        if count == 0:
            continue
        points = coords[index, :count].round().long()
        z = points[:, 0].clamp(0, spatial[0] - 1)
        y = points[:, 1].clamp(0, spatial[1] - 1)
        x = points[:, 2].clamp(0, spatial[2] - 1)
        target[index, z, y, x] = 1.0
    return target


def precision_focal_detection_loss(
    det_logits: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    neg_weight: float = 0.25,
    *,
    hard_negative_ratio: int = 48,
    gamma: float = 2.0,
    ignore_kernel: tuple[int, int, int] = (3, 5, 5),
) -> torch.Tensor:
    """Focal positive loss plus top-k hard negatives away from annotated cells.

    Sparse annotations make it unsafe to label every non-annotated voxel as a
    definite negative.  This loss ignores the immediate anisotropic
    neighbourhood of each annotation, then learns from only the most confident
    far-away false positives.  It therefore targets the candidate-pool problem
    documented in ``problematica_hipotesis_y_conclusion.md`` while preserving
    gradients for every known positive.

    The call signature is compatible with the starter kit's detection loss so
    it can be passed as an explicit training hook.
    """

    if det_logits.ndim != 5 or det_logits.shape[1] != 1:
        raise ValueError("det_logits must have shape (B, 1, Z, Y, X)")
    if hard_negative_ratio < 1:
        raise ValueError("hard_negative_ratio must be positive")

    logits = det_logits[:, 0]
    target = _point_target(det_logits, coords, mask)
    probabilities = torch.sigmoid(logits)

    positive_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_focal = ((1.0 - probabilities) ** gamma) * positive_bce

    kernel = tuple(int(value) for value in ignore_kernel)
    if any(value < 1 or value % 2 == 0 for value in kernel):
        raise ValueError("ignore_kernel values must be positive odd integers")
    padding = tuple(value // 2 for value in kernel)
    near_positive = F.max_pool3d(
        target.unsqueeze(1), kernel_size=kernel, stride=1, padding=padding
    )[:, 0].bool()

    losses = []
    for index in range(logits.shape[0]):
        positive_mask = target[index].bool()
        n_positive = int(positive_mask.sum().item())
        if n_positive:
            positive_term = positive_focal[index][positive_mask].mean()
        else:
            positive_term = logits[index].sum() * 0.0

        negative_mask = ~near_positive[index]
        negative_logits = logits[index][negative_mask]
        if negative_logits.numel():
            negative_bce = F.binary_cross_entropy_with_logits(
                negative_logits,
                torch.zeros_like(negative_logits),
                reduction="none",
            )
            negative_focal = torch.sigmoid(negative_logits).pow(gamma) * negative_bce
            keep = min(
                negative_focal.numel(),
                max(hard_negative_ratio, hard_negative_ratio * max(n_positive, 1)),
            )
            negative_term = torch.topk(negative_focal, k=keep, sorted=False).values.mean()
        else:
            negative_term = logits[index].sum() * 0.0

        losses.append(positive_term + float(neg_weight) * negative_term)

    return torch.stack(losses).mean()
