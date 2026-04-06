import pytest
import torch

from ir_torch.nn.loss.pointwise import PointwiseKLDivergenceLoss, PointwiseMSELoss


class TestPointwiseMSELoss:
    def test_zero_loss(self):
        logits = torch.tensor([[[2.0], [3.0]]])
        labels = torch.tensor([[[2.0], [3.0]]])
        loss, _ = PointwiseMSELoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0)

    def test_known_value(self):
        logits = torch.tensor([[[1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [4.0]]])
        loss, _ = PointwiseMSELoss()(logits, labels)
        # (1-3)^2 + (2-4)^2 = 4+4 = 8, mean = 4.0
        assert loss.item() == pytest.approx(4.0)

    def test_reduction_sum(self):
        logits = torch.tensor([[[1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [4.0]]])
        loss, _ = PointwiseMSELoss(reduction="sum")(logits, labels)
        assert loss.item() == pytest.approx(8.0)

    def test_reduction_none(self):
        logits = torch.tensor([[[1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [4.0]]])
        loss, _ = PointwiseMSELoss(reduction="none")(logits, labels)
        assert loss.shape == (1, 2)

    def test_with_item_mask(self):
        logits = torch.tensor([[[1.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [4.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss, _ = PointwiseMSELoss()(logits, labels, item_mask=mask)
        # Only first two items: (4+4)/2 = 4.0
        assert loss.item() == pytest.approx(4.0)

    def test_item_mask_zeroes_padded(self):
        logits = torch.tensor([[[1.0], [2.0], [99.0]]])
        labels = torch.tensor([[[1.0], [2.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss, _ = PointwiseMSELoss()(logits, labels, item_mask=mask)
        # Padded item should not contribute
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        logits = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [4.0]]])
        loss, _ = PointwiseMSELoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None


class TestPointwiseKLDivergenceLoss:
    def test_perfect_match_low_loss(self):
        # logits that produce a distribution matching the target should yield ~0 loss
        labels = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]])
        logits = torch.log(labels)  # if softmax(log(p))=p, KL=0
        loss, _ = PointwiseKLDivergenceLoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_loss(self):
        logits = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        labels = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        loss, _ = PointwiseKLDivergenceLoss()(logits, labels)
        assert loss.item() > 0.0

    def test_reduction_none(self):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        loss, _ = PointwiseKLDivergenceLoss(reduction="none")(logits, labels)
        assert loss.shape == (1, 2)

    def test_with_item_mask(self):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, -5.0]]])
        labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss_masked, _ = PointwiseKLDivergenceLoss()(logits, labels, item_mask=mask)
        loss_unmasked, _ = PointwiseKLDivergenceLoss()(logits[:, :2], labels[:, :2])
        assert loss_masked.item() == pytest.approx(loss_unmasked.item(), abs=1e-5)

    def test_gradient_flows(self):
        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        labels = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
        loss, _ = PointwiseKLDivergenceLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_unnormalised_labels(self):
        logits = torch.tensor([[[1.0, 0.0]]])
        labels_raw = torch.tensor([[[2.0, 8.0]]])
        labels_norm = torch.tensor([[[0.2, 0.8]]])
        loss_raw, _ = PointwiseKLDivergenceLoss()(logits, labels_raw)
        loss_norm, _ = PointwiseKLDivergenceLoss()(logits, labels_norm)
        assert loss_raw.item() == pytest.approx(loss_norm.item(), abs=1e-5)
