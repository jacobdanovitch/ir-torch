from __future__ import annotations

import torch
import torch.nn as nn


class MSEMarginLoss(nn.Module):
    """Pairwise MSE-margin loss.

    For every ``(i, j)`` pair within each query where ``label_i > label_j``,
    minimises the squared difference between the score margin
    ``(s_i - s_j)`` and the label margin ``(y_i - y_j)``.

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
        y = labels.squeeze(-1)  # (batch, items)

        s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (batch, items, items)
        y_diff = y.unsqueeze(2) - y.unsqueeze(1)  # (batch, items, items)

        pair_mask = y_diff > 0

        if item_mask is not None:
            valid = item_mask.unsqueeze(2) & item_mask.unsqueeze(1)
            pair_mask = pair_mask & valid

        per_pair = (s_diff - y_diff).pow(2) * pair_mask

        if self.reduction == "none":
            counts = pair_mask.sum(dim=(1, 2)).clamp(min=1)
            return per_pair.sum(dim=(1, 2)) / counts, None
        if self.reduction == "sum":
            return per_pair.sum(), None

        total_pairs = pair_mask.sum().clamp(min=1)
        return per_pair.sum() / total_pairs, None
