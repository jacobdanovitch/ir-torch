from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxRankMSELoss(nn.Module):
    """Approximate Rank-Discounted MSE loss (Schlatt et al., 2025).

    Computes the MSE between approximate (differentiable) ranks derived from
    the predicted scores and the true ranks derived from the ground-truth
    labels, optionally weighted by a position-based discount.

    Approximate ranks are obtained by summing pairwise sigmoid comparisons::

        approx_rank_i = 1 + sum_j sigmoid((s_j - s_i) / temperature)

    Args:
        temperature: Controls the smoothness of the rank approximation.
            Lower values give sharper (closer to true) ranks but noisier
            gradients.
        discount: Position-based weighting applied to per-item MSE.
            ``'log2'`` uses ``1 / log2(rank + 1)`` (NDCG-style),
            ``'reciprocal'`` uses ``1 / rank`` (MRR-style),
            ``None`` applies uniform weighting.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``, returns ``(batch,)``)

    Reference:
        Schlatt et al., *Rank-DistiLLM: Closing the Effectiveness Gap Between
        Cross-Encoders and LLMs for Passage Re-ranking*, ECIR 2025.

    Source:
        https://github.com/webis-de/lightning-ir/blob/main/lightning_ir/loss/approximate.py
    """

    def __init__(
        self,
        temperature: float = 1.0,
        discount: Literal["log2", "reciprocal"] | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.discount = discount
        self.reduction = reduction

    @staticmethod
    def _approx_ranks(scores: torch.Tensor, temperature: float) -> torch.Tensor:
        """Differentiable rank approximation via pairwise sigmoid."""
        # score_diff[b, i, j] = s_j - s_i  (positive when j beats i)
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        probs = torch.sigmoid(score_diff / temperature)
        # Zero out the diagonal (item vs itself)
        eye = torch.eye(scores.shape[1], device=scores.device, dtype=scores.dtype)
        probs = probs * (1 - eye)
        return probs.sum(dim=-1) + 1  # (batch, items)

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

        approx_ranks = self._approx_ranks(scores, self.temperature)

        # True ranks from labels (descending: highest label = rank 1)
        true_ranks = torch.argsort(torch.argsort(y, descending=True)) + 1
        true_ranks = true_ranks.to(approx_ranks)

        per_item = F.mse_loss(approx_ranks, true_ranks, reduction="none")

        # Position-based discount (based on true rank)
        if self.discount == "log2":
            weight = 1.0 / torch.log2(true_ranks + 1)
        elif self.discount == "reciprocal":
            weight = 1.0 / true_ranks
        else:
            weight = torch.ones_like(per_item)

        per_item = per_item * weight

        if item_mask is not None:
            per_item = per_item.masked_fill(~item_mask, 0.0)
            counts = item_mask.sum(dim=-1).clamp(min=1)
            per_query = per_item.sum(dim=-1) / counts
        else:
            per_query = per_item.mean(dim=-1)

        if self.reduction == "mean":
            return per_query.mean(), None
        if self.reduction == "sum":
            return per_query.sum(), None
        return per_query, None
