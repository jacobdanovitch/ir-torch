import pytest
import torch

from ir_torch.nn.loss.listwise import (
    ARPWeighting,
    LambdaRankLoss,
    ListNetLoss,
    ListwiseKLDivergenceLoss,
    MRRWeighting,
    NDCGWeighting,
    RCRLoss,
)

# ---------------------------------------------------------------------------
# ListNet
# ---------------------------------------------------------------------------


class TestListNetLoss:
    def test_perfect_ranking_low_loss(self):
        # Scores perfectly mirror labels -> low (but not zero) cross-entropy
        logits = torch.tensor([[[3.0], [2.0], [1.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = ListNetLoss()(logits, labels)
        assert loss.item() >= 0.0

    def test_reversed_ranking_higher_loss(self):
        logits_good = torch.tensor([[[3.0], [2.0], [1.0]]])
        logits_bad = torch.tensor([[[1.0], [2.0], [3.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss_good = ListNetLoss()(logits_good, labels)
        loss_bad = ListNetLoss()(logits_bad, labels)
        assert loss_bad.item() > loss_good.item()

    def test_reduction_none(self):
        logits = torch.tensor([[[3.0], [1.0]], [[1.0], [3.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        loss = ListNetLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_with_item_mask(self):
        logits = torch.tensor([[[3.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [2.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss_masked = ListNetLoss()(logits, labels, item_mask=mask)
        loss_no_pad = ListNetLoss()(logits[:, :2], labels[:, :2])
        assert loss_masked.item() == pytest.approx(loss_no_pad.item(), abs=1e-4)

    def test_gradient_flows(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = ListNetLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# LambdaRank
# ---------------------------------------------------------------------------


class TestLambdaRankLoss:
    def test_perfect_ranking_zero_ish_loss(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = LambdaRankLoss()(logits, labels)
        assert loss.item() >= 0.0

    def test_wrong_ranking_higher_loss(self):
        logits_good = torch.tensor([[[3.0], [2.0], [1.0]]])
        logits_bad = torch.tensor([[[1.0], [2.0], [3.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss_good = LambdaRankLoss()(logits_good, labels)
        loss_bad = LambdaRankLoss()(logits_bad, labels)
        assert loss_bad.item() > loss_good.item()

    def test_ndcg_weighting_default(self):
        logits = torch.tensor([[[3.0], [1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0], [1.0]]])
        loss = LambdaRankLoss(weighting=NDCGWeighting())(logits, labels)
        assert loss.item() >= 0.0

    def test_ndcg_weighting_with_k(self):
        logits = torch.tensor([[[3.0], [1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0], [1.0]]])
        loss = LambdaRankLoss(weighting=NDCGWeighting(k=2))(logits, labels)
        assert loss.item() >= 0.0

    def test_mrr_weighting(self):
        logits = torch.tensor([[[3.0], [1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0], [1.0]]])
        loss = LambdaRankLoss(weighting=MRRWeighting())(logits, labels)
        assert loss.item() >= 0.0

    def test_arp_weighting(self):
        logits = torch.tensor([[[3.0], [1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0], [1.0]]])
        loss = LambdaRankLoss(weighting=ARPWeighting())(logits, labels)
        assert loss.item() >= 0.0

    def test_reduction_none(self):
        logits = torch.tensor([[[3.0], [1.0]], [[1.0], [3.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        loss = LambdaRankLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_with_item_mask(self):
        logits = torch.tensor([[[3.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [1.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss = LambdaRankLoss()(logits, labels, item_mask=mask)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = LambdaRankLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_sigma_scaling(self):
        logits = torch.tensor([[[3.0], [1.0], [2.0]]])
        labels = torch.tensor([[[3.0], [0.0], [1.0]]])
        loss1 = LambdaRankLoss(sigma=1.0)(logits, labels)
        loss2 = LambdaRankLoss(sigma=2.0)(logits, labels)
        assert loss1.item() != pytest.approx(loss2.item(), abs=1e-3)


# ---------------------------------------------------------------------------
# Listwise KL Divergence
# ---------------------------------------------------------------------------


class TestListwiseKLDivergenceLoss:
    def test_matching_distribution_low_loss(self):
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        logits = labels.clone()
        loss = ListwiseKLDivergenceLoss()(logits, labels)
        # KL(normalised_labels || softmax(logits)) is low but not exactly 0
        # because normalise(y) != softmax(y)
        assert loss.item() < 0.1

    def test_different_distribution_positive_loss(self):
        logits = torch.tensor([[[1.0], [2.0], [3.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = ListwiseKLDivergenceLoss()(logits, labels)
        assert loss.item() > 0.0

    def test_reduction_none(self):
        logits = torch.tensor([[[3.0], [1.0]], [[1.0], [3.0]]])
        labels = torch.tensor([[[1.0], [0.5]], [[0.5], [1.0]]])
        loss = ListwiseKLDivergenceLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_with_item_mask(self):
        logits = torch.tensor([[[3.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [2.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss_masked = ListwiseKLDivergenceLoss()(logits, labels, item_mask=mask)
        loss_no_pad = ListwiseKLDivergenceLoss()(logits[:, :2], labels[:, :2])
        assert loss_masked.item() == pytest.approx(loss_no_pad.item(), abs=1e-4)

    def test_gradient_flows(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = ListwiseKLDivergenceLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# RCR
# ---------------------------------------------------------------------------


class TestRCRLoss:
    def test_alpha_zero_equals_listnet(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        rcr = RCRLoss(alpha=0.0)(logits, labels)
        listnet = ListNetLoss()(logits, labels)
        assert rcr.item() == pytest.approx(listnet.item(), abs=1e-5)

    def test_alpha_one_equals_mse(self):
        logits = torch.tensor([[[1.0], [2.0], [3.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        rcr = RCRLoss(alpha=1.0)(logits, labels)
        # MSE mean over items: ((1-3)^2 + (2-2)^2 + (3-1)^2)/3 = 8/3
        expected = torch.tensor(8.0 / 3.0)
        assert rcr.item() == pytest.approx(expected.item(), abs=1e-4)

    def test_default_alpha(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]])
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = RCRLoss()(logits, labels)
        assert loss.item() >= 0.0

    def test_reduction_none(self):
        logits = torch.tensor([[[3.0], [1.0]], [[1.0], [3.0]]])
        labels = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
        loss = RCRLoss(reduction="none")(logits, labels)
        assert loss.shape == (2,)

    def test_with_item_mask(self):
        logits = torch.tensor([[[3.0], [2.0], [0.0]]])
        labels = torch.tensor([[[3.0], [2.0], [0.0]]])
        mask = torch.tensor([[True, True, False]])
        loss = RCRLoss()(logits, labels, item_mask=mask)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        logits = torch.tensor([[[3.0], [2.0], [1.0]]], requires_grad=True)
        labels = torch.tensor([[[3.0], [2.0], [1.0]]])
        loss = RCRLoss()(logits, labels)
        loss.backward()
        assert logits.grad is not None
