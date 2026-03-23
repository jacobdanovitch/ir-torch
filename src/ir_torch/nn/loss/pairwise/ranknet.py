from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    """RankNet pairwise loss (Burges et al., 2005).

    Computes the binary cross-entropy on the predicted probability that the
    first item is ranked higher than the second.

    Expects each example to have exactly 2 items: ``items[:, 0]`` is the
    positive (higher-relevance) item and ``items[:, 1]`` is the negative.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        sigma: Scaling factor for the logit difference (default 1.0).

    Shapes:
        - logits: ``(batch, 2, 1)``
        - labels: ``(batch, 2, 1)`` — used to derive pairwise target
        - output: scalar (unless ``reduction='none'``)
    """

    def __init__(self, reduction: str = "mean", sigma: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.sigma = sigma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s_i = logits[:, 0, 0]  # (batch,)
        s_j = logits[:, 1, 0]  # (batch,)

        y_i = labels[:, 0, 0]
        y_j = labels[:, 1, 0]

        # Target: P(i > j) = 1 if y_i > y_j, 0.5 if equal, 0 otherwise
        target = 0.5 * (1.0 + (y_i - y_j).sign())

        loss = F.binary_cross_entropy_with_logits(self.sigma * (s_i - s_j), target, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
