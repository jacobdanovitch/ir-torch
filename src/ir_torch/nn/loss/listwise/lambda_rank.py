from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------


class LambdaWeighting(ABC):
    """Base class for LambdaRank weighting schemes."""

    @abstractmethod
    def weight(
        self,
        labels: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-swap weight deltas.

        Args:
            labels: ``(batch, items)`` relevance labels.
            sorted_indices: ``(batch, items)`` indices that sort items by
                predicted score (descending).

        Returns:
            ``(batch, items, items)`` pairwise weight matrix.
        """


class NDCGWeighting(LambdaWeighting):
    """Weight swaps by the absolute change in NDCG (Burges et al., 2006)."""

    def __init__(self, k: int | None = None):
        self.k = k

    def weight(
        self,
        labels: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, n = labels.shape
        device = labels.device

        gains = (2.0**labels) - 1.0  # (batch, items)

        # Ideal DCG
        ideal_sorted = gains.sort(dim=-1, descending=True).values
        positions = torch.arange(1, n + 1, device=device, dtype=labels.dtype)
        discounts = 1.0 / torch.log2(positions + 1.0)

        if self.k is not None:
            discounts[self.k :] = 0.0

        idcg = (ideal_sorted * discounts).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (batch, 1)

        # Ranks from predicted order
        ranks = torch.zeros_like(sorted_indices, dtype=labels.dtype)
        ranks.scatter_(1, sorted_indices, positions.unsqueeze(0).expand(batch, -1))

        # |delta NDCG| for swapping items i, j
        disc_i = 1.0 / torch.log2(ranks + 1.0)  # (batch, items)
        gain_diff = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()  # (batch, i, j)
        disc_diff = (disc_i.unsqueeze(2) - disc_i.unsqueeze(1)).abs()  # (batch, i, j)

        return gain_diff * disc_diff / idcg.unsqueeze(-1)


class MRRWeighting(LambdaWeighting):
    """Weight swaps by the absolute change in MRR."""

    def weight(
        self,
        labels: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, n = labels.shape
        device = labels.device

        positions = torch.arange(1, n + 1, device=device, dtype=labels.dtype)
        ranks = torch.zeros_like(sorted_indices, dtype=labels.dtype)
        ranks.scatter_(1, sorted_indices, positions.unsqueeze(0).expand(batch, -1))

        rr = 1.0 / ranks  # (batch, items)
        return (rr.unsqueeze(2) - rr.unsqueeze(1)).abs()


class ARPWeighting(LambdaWeighting):
    """Weight swaps by the absolute change in Average Rank Position."""

    def weight(
        self,
        labels: torch.Tensor,
        sorted_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, n = labels.shape
        device = labels.device

        positions = torch.arange(1, n + 1, device=device, dtype=labels.dtype)
        ranks = torch.zeros_like(sorted_indices, dtype=labels.dtype)
        ranks.scatter_(1, sorted_indices, positions.unsqueeze(0).expand(batch, -1))

        # ARP weight: relevant items should be pushed up
        gain_diff = (labels.unsqueeze(2) - labels.unsqueeze(1)).abs()
        rank_diff = (ranks.unsqueeze(2) - ranks.unsqueeze(1)).abs()
        return gain_diff * rank_diff


# ---------------------------------------------------------------------------
# LambdaRank loss
# ---------------------------------------------------------------------------


class LambdaRankLoss(nn.Module):
    """LambdaRank loss (Burges et al., 2006).

    Computes RankNet-style pairwise logistic losses weighted by a
    position-aware metric delta (|delta NDCG|, |delta MRR|, etc.).

    Args:
        weighting: A :class:`LambdaWeighting` instance.
        sigma: Scaling factor for score differences (default 1.0).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: scalar (unless ``reduction='none'``, returns ``(batch,)``)
    """

    def __init__(
        self,
        weighting: LambdaWeighting | None = None,
        sigma: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weighting = weighting or NDCGWeighting()
        self.sigma = sigma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        scores = logits.squeeze(-1)  # (batch, items)
        y = labels.squeeze(-1).float()  # (batch, items)

        # Sort by predicted score (descending) to get current ranking
        sorted_indices = scores.detach().argsort(dim=-1, descending=True)

        # Pairwise score differences: s_i - s_j
        s_ij = self.sigma * (scores.unsqueeze(2) - scores.unsqueeze(1))  # (batch, i, j)

        # Pairwise label sign: 1 if y_i > y_j, 0 if equal, -1 otherwise
        y_ij = (y.unsqueeze(2) - y.unsqueeze(1)).sign()  # (batch, i, j)

        # Metric delta weights
        weights = self.weighting.weight(y, sorted_indices)  # (batch, i, j)

        # Lambda: logistic loss * weight, only upper triangle (i < j) to avoid double-counting
        pair_loss = torch.log1p(torch.exp(-y_ij * s_ij)) * weights

        # Mask lower triangle + diagonal
        n = scores.shape[1]
        triu_mask = torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), diagonal=1)
        pair_loss = pair_loss * triu_mask.unsqueeze(0)

        if item_mask is not None:
            # Zero out pairs involving padded items
            valid = item_mask.unsqueeze(2) & item_mask.unsqueeze(1)  # (batch, i, j)
            pair_loss = pair_loss * valid

        loss = pair_loss.sum(dim=(1, 2))  # (batch,)

        if self.reduction == "mean":
            return loss.mean(), None
        if self.reduction == "sum":
            return loss.sum(), None
        return loss, None
