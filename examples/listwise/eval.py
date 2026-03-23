"""Evaluate a trained listwise ranker with NDCG on a test set.

Usage
-----
.. code-block:: bash

    python examples/listwise/eval.py \
        --model_path /tmp/listwise_ranker \
        --test_path /tmp/test.jsonl \
        --max_length 128 \
        --batch_size 256
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ir_torch.data import RankingCollator, RankingDataset

# ──────────────────────────────────────────────────────────────────────────────
# NDCG computation
# ──────────────────────────────────────────────────────────────────────────────


def _dcg(relevances: torch.Tensor, k: int) -> torch.Tensor:
    """Compute DCG@k for each query in a batch.

    Args:
        relevances: ``(batch, items)`` relevance labels in ranked order.
        k: Cutoff.

    Returns:
        ``(batch,)`` DCG values.
    """
    relevances = relevances[:, :k].float()
    positions = torch.arange(1, relevances.shape[1] + 1, device=relevances.device).float()
    discounts = torch.log2(positions + 1)
    return (relevances / discounts).sum(dim=-1)


def ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Compute NDCG@k for each query in a batch.

    Args:
        scores: ``(batch, items)`` predicted scores.
        labels: ``(batch, items)`` relevance labels.
        k: Cutoff.

    Returns:
        ``(batch,)`` NDCG values.  Queries with no relevant items get 0.
    """
    # Sort items by predicted score (descending)
    _, pred_order = scores.sort(dim=-1, descending=True)
    ranked_labels = labels.gather(dim=-1, index=pred_order)

    dcg = _dcg(ranked_labels, k)

    # Ideal ranking: sort by true labels
    ideal_labels, _ = labels.sort(dim=-1, descending=True)
    idcg = _dcg(ideal_labels, k)

    # Avoid division by zero for queries with no relevant items
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    ks: list[int],
    device: torch.device,
) -> dict[int, float]:
    """Run evaluation and return mean NDCG at each cutoff."""
    model.eval()

    all_ndcg: dict[int, list[torch.Tensor]] = {k: [] for k in ks}
    total = 0

    for batch in loader:
        batch = batch.to(device)
        batch_size, num_items, seq_len = batch.input_ids.shape

        input_ids = batch.input_ids.view(-1, seq_len)
        attention_mask = batch.attention_mask.view(-1, seq_len)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        scores = logits.view(batch_size, num_items)  # (batch, items)
        labels = batch.labels.squeeze(-1)  # (batch, items)

        # Mask out padded items by giving them -inf score
        if batch.item_mask is not None:
            scores = scores.masked_fill(~batch.item_mask, float("-inf"))

        for k in ks:
            all_ndcg[k].append(ndcg(scores, labels, k))

        total += batch_size

    results = {}
    for k in ks:
        results[k] = torch.cat(all_ndcg[k]).mean().item()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a listwise ranker (NDCG)")
    p.add_argument("--model_path", type=str, required=True, help="Path to saved model directory")
    p.add_argument("--test_path", type=str, required=True, help="Path to test JSONL file or directory")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_items", type=int, default=None, help="Max items per query (splits larger queries)")
    p.add_argument(
        "--template",
        type=str,
        default="Query: {query} Document: {text} Relevant:",
        help="Prompt template (must match training)",
    )
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10], help="NDCG cutoffs")
    p.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(device)

    collator = RankingCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        template=args.template,
    )

    test_ds = RankingDataset(args.test_path, max_items=args.max_items)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
    )

    print(f"Evaluating {args.model_path} on {args.test_path} ({len(test_ds)} queries)")
    print(f"Device: {device}")

    results = evaluate(model, test_loader, args.ks, device)

    print()
    for k, v in sorted(results.items()):
        print(f"  NDCG@{k}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
