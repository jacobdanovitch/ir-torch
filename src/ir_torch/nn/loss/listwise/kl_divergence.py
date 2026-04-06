from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListwiseKLDivergenceLoss(nn.Module):
    """Listwise KL-divergence loss for single-class relevance labels.

    Treats each row of labels across items as an (un-normalised) target
    probability distribution and computes the KL divergence between the
    normalised target distribution and the softmax of the predicted scores.

    This is useful when labels represent graded relevance and you want
    the score distribution to match the label distribution.

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
            y = y.masked_fill(~item_mask, 0.0)

        # Normalise labels to a probability distribution per query
        target = y / y.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        log_probs = F.log_softmax(scores, dim=-1)

        # KL per query: sum_i target_i * (log target_i - log_probs_i)
        loss = F.kl_div(log_probs, target, reduction="none").sum(dim=-1)  # (batch,)

        if self.reduction == "mean":
            return loss.mean(), None
        if self.reduction == "sum":
            return loss.sum(), None
        return loss, None
