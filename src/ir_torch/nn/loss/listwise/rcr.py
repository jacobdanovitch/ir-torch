from __future__ import annotations

import torch

from ir_torch.nn.loss.listwise.listnet import ListNetLoss
from ir_torch.nn.loss.multitask import WeightedMultiTaskLoss
from ir_torch.nn.loss.pointwise.mse import PointwiseMSELoss


class RCRLoss(WeightedMultiTaskLoss):
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
        - output: ``(scalar, {"mse": ..., "listnet": ...})`` (unless ``reduction='none'``)
    """

    def __init__(self, alpha: float = 0.5, reduction: str = "mean"):
        # Sub-losses always use reduction="none" so we get per-query values
        # and handle final reduction here (MSE returns (batch, items) which
        # we reduce to (batch,) to match ListNet's (batch,) shape).
        super().__init__({
            "mse": (alpha, PointwiseMSELoss(reduction="none")),
            "listnet": (1.0 - alpha, ListNetLoss(reduction="none")),
        })
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        # Get per-item MSE and per-query ListNet
        mse_result = self.mse(logits, labels, item_mask=item_mask)
        mse_per_item = mse_result[0] if isinstance(mse_result, tuple) else mse_result

        listnet_result = self.listnet(logits, labels, item_mask=item_mask)
        listnet_per_query = listnet_result[0] if isinstance(listnet_result, tuple) else listnet_result

        # Reduce MSE from (batch, items) to (batch,) to match ListNet
        if item_mask is not None:
            mse_per_query = mse_per_item.sum(dim=-1) / item_mask.sum(dim=-1).clamp(min=1)
        else:
            mse_per_query = mse_per_item.mean(dim=-1)

        alpha = self._weights["mse"]
        beta = self._weights["listnet"]
        per_query = alpha * mse_per_query + beta * listnet_per_query

        sub_losses: dict[str, torch.Tensor] = {
            "mse": mse_per_query.detach().mean(),
            "listnet": listnet_per_query.detach().mean(),
        }

        if self.reduction == "mean":
            return per_query.mean(), sub_losses
        if self.reduction == "sum":
            return per_query.sum(), sub_losses
        return per_query, sub_losses
