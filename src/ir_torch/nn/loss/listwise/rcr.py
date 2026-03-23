from __future__ import annotations

import torch
import torch.nn as nn


class RCRLoss(nn.Module):
    """Regression Compatible Ranking (RCR) loss (Busa-Fekete et al., 2021).

    Combines a pointwise regression objective (MSE) with a listwise ranking
    component that penalises mis-orderings through a softmax cross-entropy
    term, giving a loss that is consistent for both regression *and* ranking.

    ``L = alpha * MSE(scores, labels) + (1 - alpha) * ListNet(scores, labels)``

    Args:
        alpha: Trade-off between regression and ranking (default 0.5).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``)
    """

    def __init__(self, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = logits.squeeze(-1)  # (batch, items)
        y = labels.squeeze(-1).float()  # (batch, items)

        # --- Pointwise MSE component ---
        mse = (scores - y).pow(2)  # (batch, items)
        if item_mask is not None:
            mse = mse * item_mask

        # --- Listwise (ListNet-style) component ---
        if item_mask is not None:
            fill = torch.finfo(scores.dtype).min
            masked_scores = scores.masked_fill(~item_mask, fill)
            masked_y = y.masked_fill(~item_mask, 0.0)
        else:
            masked_scores = scores
            masked_y = y

        p_y = torch.softmax(masked_y, dim=-1)
        log_p_s = torch.log_softmax(masked_scores, dim=-1)
        listnet = -(p_y * log_p_s).sum(dim=-1)  # (batch,)

        # --- Combine ---
        if self.reduction == "none":
            # Return per-query loss
            if item_mask is not None:
                mse_per_query = mse.sum(dim=-1) / item_mask.sum(dim=-1).clamp(min=1)
            else:
                mse_per_query = mse.mean(dim=-1)
            return self.alpha * mse_per_query + (1.0 - self.alpha) * listnet

        if self.reduction == "mean":
            mse_loss = mse.sum() / item_mask.sum().clamp(min=1) if item_mask is not None else mse.mean()
            listnet_loss = listnet.mean()
        else:  # sum
            mse_loss = mse.sum()
            listnet_loss = listnet.sum()

        return self.alpha * mse_loss + (1.0 - self.alpha) * listnet_loss
