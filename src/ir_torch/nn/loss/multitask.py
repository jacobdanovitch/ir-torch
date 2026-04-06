from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Abstract base class for multi-task ranking losses.

    A multi-task loss composes several sub-losses and returns both the
    combined scalar loss and a dictionary of named sub-loss values for
    logging.

    Subclasses must implement :meth:`forward`.

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: ``(scalar, {"name": scalar, ...})``
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        raise NotImplementedError


class WeightedMultiTaskLoss(MultiTaskLoss):
    """Weighted sum of an arbitrary number of sub-losses.

    Each sub-loss is called with the same ``(logits, labels, item_mask)``
    arguments. The total loss is ``sum(weight_i * loss_i)``.

    Args:
        losses: Mapping from name to ``(weight, loss_module)`` pairs.

    Example::

        criterion = WeightedMultiTaskLoss({
            "mse": (0.5, PointwiseMSELoss()),
            "listnet": (0.5, ListNetLoss()),
        })
        loss, sub_losses = criterion(logits, labels)
        # sub_losses == {"mse": ..., "listnet": ...}
    """

    def __init__(self, losses: dict[str, tuple[float, nn.Module]]):
        super().__init__()
        if not losses:
            msg = "WeightedMultiTaskLoss requires at least one sub-loss"
            raise ValueError(msg)
        self._weights: dict[str, float] = {}
        for name, (weight, module) in losses.items():
            self._weights[name] = weight
            self.add_module(name, module)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        sub_losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for name, weight in self._weights.items():
            module = getattr(self, name)
            result = module(logits, labels, item_mask=item_mask)
            loss_val = result[0] if isinstance(result, tuple) else result
            sub_losses[name] = loss_val.detach()
            total = total + weight * loss_val

        return total, sub_losses


class CalibratedListwiseLoss(WeightedMultiTaskLoss):
    """Listwise loss calibrated by a pointwise regression term.

    Combines one pointwise loss (per-item, e.g. MSE) with one listwise loss
    (per-query, e.g. ListNet) using a mixing coefficient ``alpha``.
    Sub-losses are always evaluated with ``reduction='none'``; the pointwise
    component is reduced from ``(batch, items)`` to ``(batch,)`` before
    combining, and the final reduction is applied here.

    ``L = alpha * pointwise(scores, labels) + (1 - alpha) * listwise(scores, labels)``

    When ``pointwise_fallback=True`` (the default), queries whose labels are
    all identical fall back to using only the pointwise loss.  Listwise losses
    produce uninformative gradients for such queries (e.g. softmax over equal
    values is uniform regardless of scores), so using only the pointwise term
    gives a meaningful training signal.

    Args:
        pointwise: A pointwise loss module (returns ``(batch, items)`` with
            ``reduction='none'``).
        listwise: A listwise loss module (returns ``(batch,)`` with
            ``reduction='none'``).
        alpha: Weight of the pointwise term (default 0.5).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
        pointwise_fallback: If ``True``, queries with uniform labels use
            only the pointwise loss (default ``True``).

    Shapes:
        - logits: ``(batch, items, 1)``
        - labels: ``(batch, items, 1)``
        - item_mask: ``(batch, items)`` or ``None``
        - output: ``(scalar, {"pointwise": ..., "listwise": ...})``
    """

    def __init__(
        self,
        pointwise: nn.Module,
        listwise: nn.Module,
        alpha: float = 0.5,
        reduction: str = "mean",
        pointwise_fallback: bool = True,
    ):
        super().__init__({
            "pointwise": (alpha, pointwise),
            "listwise": (1.0 - alpha, listwise),
        })
        self.reduction = reduction
        self.pointwise_fallback = pointwise_fallback

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        pw_result = self.pointwise(logits, labels, item_mask=item_mask)
        pw_per_item = pw_result[0] if isinstance(pw_result, tuple) else pw_result

        lw_result = self.listwise(logits, labels, item_mask=item_mask)
        lw_per_query = lw_result[0] if isinstance(lw_result, tuple) else lw_result

        # Reduce pointwise from (batch, items) → (batch,)
        if item_mask is not None:
            pw_per_query = pw_per_item.sum(dim=-1) / item_mask.sum(dim=-1).clamp(min=1)
        else:
            pw_per_query = pw_per_item.mean(dim=-1)

        alpha = self._weights["pointwise"]
        beta = self._weights["listwise"]
        per_query = alpha * pw_per_query + beta * lw_per_query

        # Fall back to pointwise-only for queries with uniform labels
        if self.pointwise_fallback:
            y = labels.squeeze(-1)  # (batch, items)
            if item_mask is not None:
                # Compare each label to the first valid label per query
                first = (y * item_mask).sum(dim=-1) / item_mask.sum(dim=-1).clamp(min=1)
                uniform = ((y - first.unsqueeze(-1)).abs() * item_mask).sum(dim=-1) == 0
            else:
                uniform = (y == y[:, :1]).all(dim=-1)  # (batch,)
            per_query = torch.where(uniform, pw_per_query, per_query)

        sub_losses: dict[str, torch.Tensor] = {
            "pointwise": pw_per_query.detach().mean(),
            "listwise": lw_per_query.detach().mean(),
        }

        if self.reduction == "mean":
            return per_query.mean(), sub_losses
        if self.reduction == "sum":
            return per_query.sum(), sub_losses
        return per_query, sub_losses
