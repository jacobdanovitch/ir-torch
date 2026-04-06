from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    """RankNet pairwise loss (Burges et al., 2005).

    Computes the binary cross-entropy for every ``(i, j)`` pair of items
    within each query where ``label_i > label_j``.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        sigma: Scaling factor for the logit difference (default 1.0).

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``, returns ``(batch,)``)
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        scores = logits.squeeze(-1)  # (batch, items)
        y = labels.squeeze(-1)  # (batch, items)

        # All (i, j) differences
        s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (batch, items, items)
        y_diff = y.unsqueeze(2) - y.unsqueeze(1)  # (batch, items, items)

        # Target: P(i > j) = 1 if y_i > y_j, 0.5 if equal, 0 otherwise
        target = 0.5 * (1.0 + y_diff.sign())

        # Only keep pairs where y_i > y_j
        pair_mask = y_diff > 0

        if item_mask is not None:
            # Both items must be valid
            valid = item_mask.unsqueeze(2) & item_mask.unsqueeze(1)  # (batch, items, items)
            pair_mask = pair_mask & valid

        per_pair = F.binary_cross_entropy_with_logits(
            self.sigma * s_diff,
            target,
            reduction="none",
        )
        per_pair = per_pair * pair_mask

        if self.reduction == "none":
            counts = pair_mask.sum(dim=(1, 2)).clamp(min=1)
            return per_pair.sum(dim=(1, 2)) / counts, None
        if self.reduction == "sum":
            return per_pair.sum(), None

        total_pairs = pair_mask.sum().clamp(min=1)
        return per_pair.sum() / total_pairs, None
