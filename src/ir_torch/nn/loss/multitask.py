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
