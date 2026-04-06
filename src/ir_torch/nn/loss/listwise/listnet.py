from __future__ import annotations

import torch
import torch.nn as nn


class ListNetLoss(nn.Module):
    """ListNet loss (Cao et al., 2007).

    Computes the cross-entropy between the top-1 probability distributions
    induced by the predicted scores and the ground-truth relevance labels.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``, returns ``(batch,)``)
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
        scores = logits.squeeze(-1)  # (batch, items)
        y = labels.squeeze(-1).float()  # (batch, items)

        if item_mask is not None:
            fill = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~item_mask, fill)
            y = y.masked_fill(~item_mask, fill)

        # Top-1 probability distributions
        p_y = torch.softmax(y, dim=-1)
        p_s = torch.log_softmax(scores, dim=-1)

        # Cross-entropy per query
        loss = -(p_y * p_s).sum(dim=-1)  # (batch,)

        if self.reduction == "mean":
            return loss.mean(), None
        if self.reduction == "sum":
            return loss.sum(), None
        return loss, None
