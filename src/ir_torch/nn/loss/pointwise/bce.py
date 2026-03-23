from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseBCELoss(nn.Module):
    """Pointwise binary cross-entropy loss for ranking.

    Applies sigmoid to the predicted logits and computes BCE against
    relevance labels normalised to [0, 1] by dividing by ``label_max``.

    Args:
        label_max: Maximum relevance grade used to normalise labels
            into [0, 1] (default 1, i.e. labels are already binary).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``)
    """

    def __init__(self, label_max: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.label_max = label_max
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.squeeze(-1)  # (batch, items)
        targets = labels.squeeze(-1).float()  # (batch, items)
        if self.label_max != 1.0:
            targets = targets / self.label_max

        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")

        if item_mask is not None:
            loss = loss * item_mask

        if self.reduction == "mean":
            if item_mask is not None:
                return loss.sum() / item_mask.sum().clamp(min=1)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
