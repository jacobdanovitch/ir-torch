from __future__ import annotations

import torch
import torch.nn as nn


class PointwiseMSELoss(nn.Module):
    """Pointwise MSE loss for ranking.

    Computes mean squared error between predicted scores and relevance labels
    independently for each item.

    Args:
        reduction: Specifies the reduction to apply: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        loss = (logits.squeeze(-1) - labels.squeeze(-1)).pow(2)

        if item_mask is not None:
            loss = loss * item_mask

        if self.reduction == "mean":
            if item_mask is not None:
                return loss.sum() / item_mask.sum().clamp(min=1), None
            return loss.mean(), None
        if self.reduction == "sum":
            return loss.sum(), None
        return loss, None
