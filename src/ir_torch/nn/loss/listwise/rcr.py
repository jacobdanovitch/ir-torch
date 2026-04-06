from __future__ import annotations

from ir_torch.nn.loss.listwise.listnet import ListNetLoss
from ir_torch.nn.loss.multitask import CalibratedListwiseLoss
from ir_torch.nn.loss.pointwise.mse import PointwiseMSELoss


class RCRLoss(CalibratedListwiseLoss):
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
        - output: ``(scalar, {"pointwise": ..., "listwise": ...})``
    """

    def __init__(self, alpha: float = 0.5, reduction: str = "mean"):
        super().__init__(
            pointwise=PointwiseMSELoss(reduction="none"),
            listwise=ListNetLoss(reduction="none"),
            alpha=alpha,
            reduction=reduction,
        )
