from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseKLDivergenceLoss(nn.Module):
    """Pointwise KL-divergence loss for multi-class relevance labels.

    Each item has a vector of class logits and a corresponding target
    distribution (one-hot or soft). The KL divergence is computed per-item
    and then aggregated across the batch.

    Args:
        reduction: Specifies the reduction to apply: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, num_classes)``
        - labels: ``(batch, items, num_classes)`` — target distributions (will be
          normalised to sum to 1 if they don't already).
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
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        targets = labels / labels.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # KL per item: sum over classes
        kl = F.kl_div(log_probs, targets, reduction="none").sum(dim=-1)  # (batch, items)

        if item_mask is not None:
            kl = kl * item_mask

        if self.reduction == "mean":
            if item_mask is not None:
                return kl.sum() / item_mask.sum().clamp(min=1)
            return kl.mean()
        if self.reduction == "sum":
            return kl.sum()
        return kl
