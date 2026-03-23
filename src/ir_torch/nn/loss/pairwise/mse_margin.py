from __future__ import annotations

import torch
import torch.nn as nn


class MSEMarginLoss(nn.Module):
    """Pairwise MSE-margin loss.

    Minimises the squared difference between the score margin
    ``(s_i - s_j)`` and the label margin ``(y_i - y_j)``.

    Expects each example to have exactly 2 items.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, 2, 1)``
        - labels: ``(batch, 2, 1)``
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
    ) -> torch.Tensor:
        score_diff = logits[:, 0, 0] - logits[:, 1, 0]  # (batch,)
        label_diff = labels[:, 0, 0] - labels[:, 1, 0]  # (batch,)

        loss = (score_diff - label_diff).pow(2)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
