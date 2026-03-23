import pytest
import torch

from ir_torch.nn.loss.pairwise import MSEMarginLoss, RankNetLoss


class TestRankNetLoss:
    def test_correct_order_low_loss(self):
        # Positive item scored much higher than negative -> low loss
        logits = torch.tensor([[[5.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss = RankNetLoss()(logits, labels)
        assert loss.item() < 0.1

    def test_wrong_order_high_loss(self):
        # Positive item scored much lower than negative -> high loss
        logits = torch.tensor([[[0.0], [5.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss = RankNetLoss()(logits, labels)
        assert loss.item() > 4.0

    def test_equal_scores_tie(self):
        logits = torch.tensor([[[0.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss = RankNetLoss()(logits, labels)
        # log(1 + exp(0)) * target component; for target=1: -log(sigmoid(0))=log(2)
        assert loss.item() == pytest.approx(0.6931, abs=1e-3)

    def test_equal_labels_cross_entropy_half(self):
        logits = torch.tensor([[[3.0], [1.0]]])
        labels = torch.tensor([[[1.0], [1.0]]])
        loss = RankNetLoss()(logits, labels)
        # target = 0.5, BCE with logits(sigma*(3-1), 0.5)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(2.0), torch.tensor(0.5))
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_sigma_scaling(self):
        logits = torch.tensor([[[2.0], [1.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss1 = RankNetLoss(sigma=1.0)(logits, labels)
        loss2 = RankNetLoss(sigma=2.0)(logits, labels)
        assert loss1.item() != pytest.approx(loss2.item(), abs=1e-3)

    def test_reduction_none(self):
        logits = torch.tensor([[[2.0], [1.0]], [[3.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        loss = RankNetLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_reduction_sum(self):
        logits = torch.tensor([[[2.0], [1.0]], [[3.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        none_loss = RankNetLoss(reduction="none")(logits, labels)
        sum_loss = RankNetLoss(reduction="sum")(logits, labels)
        assert sum_loss.item() == pytest.approx(none_loss.sum().item(), abs=1e-5)

    def test_gradient_flows(self):
        logits = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
        labels = torch.tensor([[[1.0], [0.0]]])
        loss = RankNetLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_batch_of_pairs(self):
        logits = torch.tensor([[[3.0], [1.0]], [[0.5], [2.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        loss = RankNetLoss()(logits, labels)
        # First pair correct, second wrong -> moderate average loss
        assert loss.item() > 0.0


class TestMSEMarginLoss:
    def test_perfect_margin(self):
        logits = torch.tensor([[[5.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0]]])
        # score diff = 3, label diff = 3 -> loss = 0
        loss = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0)

    def test_known_value(self):
        logits = torch.tensor([[[4.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0]]])
        # score diff = 2, label diff = 3, (2-3)^2 = 1
        loss = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(1.0)

    def test_reduction_sum(self):
        logits = torch.tensor([[[4.0], [2.0]], [[1.0], [1.0]]])
        labels = torch.tensor([[[3.0], [0.0]], [[2.0], [0.0]]])
        loss_none = MSEMarginLoss(reduction="none")(logits, labels)
        loss_sum = MSEMarginLoss(reduction="sum")(logits, labels)
        assert loss_sum.item() == pytest.approx(loss_none.sum().item(), abs=1e-5)

    def test_reduction_none(self):
        logits = torch.tensor([[[4.0], [2.0]], [[1.0], [1.0]]])
        labels = torch.tensor([[[3.0], [0.0]], [[2.0], [0.0]]])
        loss = MSEMarginLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_gradient_flows(self):
        logits = torch.tensor([[[4.0], [2.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [0.0]]])
        loss = MSEMarginLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None
