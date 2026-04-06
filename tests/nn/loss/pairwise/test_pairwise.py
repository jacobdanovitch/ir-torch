import pytest
import torch

from ir_torch.nn.loss.pairwise import MSEMarginLoss, RankNetLoss


class TestRankNetLoss:
    def test_correct_order_low_loss(self):
        # Item 0 scored much higher and more relevant -> low loss
        logits = torch.tensor([[[5.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        assert loss.item() < 0.1

    def test_wrong_order_high_loss(self):
        # Item 0 is more relevant but scored lower -> high loss
        logits = torch.tensor([[[0.0], [5.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        assert loss.item() > 4.0

    def test_equal_scores_binary_pair(self):
        logits = torch.tensor([[[0.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        # One pair (0>1): target=1, BCE(0, 1) = log(2)
        assert loss.item() == pytest.approx(0.6931, abs=1e-3)

    def test_equal_labels_no_pairs(self):
        # No pairs where label_i > label_j -> loss is 0
        logits = torch.tensor([[[3.0], [1.0]]])
        labels = torch.tensor([[[1.0], [1.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_sigma_scaling(self):
        logits = torch.tensor([[[2.0], [1.0]]])
        labels = torch.tensor([[[1.0], [0.0]]])
        loss1, _ = RankNetLoss(sigma=1.0)(logits, labels)
        loss2, _ = RankNetLoss(sigma=2.0)(logits, labels)
        assert loss1.item() != pytest.approx(loss2.item(), abs=1e-3)

    def test_multi_item_query(self):
        # 3 items: labels [3, 1, 0] -> pairs (0,1), (0,2), (1,2)
        logits = torch.tensor([[[5.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        assert loss.item() < 0.5  # All correctly ordered

    def test_multi_item_wrong_order(self):
        # 3 items: labels [3, 1, 0] but scores are inverted
        logits = torch.tensor([[[0.0], [2.0], [5.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        assert loss.item() > 2.0

    def test_reduction_none(self):
        logits = torch.tensor([[[2.0], [1.0], [0.0]], [[3.0], [0.0], [1.0]]])
        labels = torch.tensor([[[2.0], [1.0], [0.0]], [[2.0], [0.0], [1.0]]])
        loss, _ = RankNetLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_reduction_sum(self):
        logits = torch.tensor([[[2.0], [1.0]], [[3.0], [0.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        _none_loss, _ = RankNetLoss(reduction="none")(logits, labels)
        sum_loss, _ = RankNetLoss(reduction="sum")(logits, labels)
        # sum reduction sums all pair losses (not per-query means)
        assert sum_loss.item() > 0.0

    def test_gradient_flows(self):
        logits = torch.tensor([[[2.0], [1.0], [0.0]]], requires_grad=True)
        labels = torch.tensor([[[2.0], [1.0], [0.0]]])
        loss, _ = RankNetLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_with_item_mask(self):
        logits = torch.tensor([[[5.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        # Mask out item 2 -> only pair (0,1)
        mask = torch.tensor([[True, True, False]])
        loss_masked, _ = RankNetLoss()(logits, labels, item_mask=mask)
        # Compare with 2-item version
        logits2 = torch.tensor([[[5.0], [2.0]]])
        labels2 = torch.tensor([[[3.0], [1.0]]])
        loss_2item, _ = RankNetLoss()(logits2, labels2)
        assert loss_masked.item() == pytest.approx(loss_2item.item(), abs=1e-5)


class TestMSEMarginLoss:
    def test_perfect_margin(self):
        logits = torch.tensor([[[5.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0]]])
        loss, _ = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0)

    def test_known_value(self):
        logits = torch.tensor([[[4.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0]]])
        # score diff = 2, label diff = 3, (2-3)^2 = 1
        loss, _ = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(1.0)

    def test_multi_item_query(self):
        # 3 items: labels [3, 1, 0], scores [5, 2, 0]
        # Pairs: (0,1): (3,1)=2, (0,2): (5,3)=2, (1,2): (2,1)=1
        # Label diffs: 2, 3, 1  Score diffs: 3, 5, 2
        # MSEs: (3-2)^2=1, (5-3)^2=4, (2-1)^2=1 -> mean = 2.0
        logits = torch.tensor([[[5.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        loss, _ = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(2.0)

    def test_equal_labels_no_pairs(self):
        logits = torch.tensor([[[4.0], [2.0]]])
        labels = torch.tensor([[[1.0], [1.0]]])
        loss, _ = MSEMarginLoss()(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_reduction_sum(self):
        logits = torch.tensor([[[4.0], [2.0]], [[1.0], [1.0]]])
        labels = torch.tensor([[[3.0], [0.0]], [[2.0], [0.0]]])
        loss_sum, _ = MSEMarginLoss(reduction="sum")(logits, labels)
        assert loss_sum.item() > 0.0

    def test_reduction_none(self):
        logits = torch.tensor([[[4.0], [2.0]], [[1.0], [1.0]]])
        labels = torch.tensor([[[3.0], [0.0]], [[2.0], [0.0]]])
        loss, _ = MSEMarginLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_gradient_flows(self):
        logits = torch.tensor([[[4.0], [2.0], [0.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        loss, _ = MSEMarginLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_with_item_mask(self):
        logits = torch.tensor([[[5.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss_masked, _ = MSEMarginLoss()(logits, labels, item_mask=mask)
        logits2 = torch.tensor([[[5.0], [2.0]]])
        labels2 = torch.tensor([[[3.0], [1.0]]])
        loss_2item, _ = MSEMarginLoss()(logits2, labels2)
        assert loss_masked.item() == pytest.approx(loss_2item.item(), abs=1e-5)
