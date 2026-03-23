"""Listwise ranking with RCR loss — PyTorch Lightning demo.

Trains the ``jhu-clsp/ettin-decoder-17m`` causal language model as a
listwise ranker using the Regression Compatible Ranking (RCR) loss
from ``ir_torch``.

Usage
-----
.. code-block:: bash

    # Install extra dependencies (only needed for this demo)
    uv pip install pytorch-lightning

    # Prepare a JSONL training file (see ``make_dummy_data`` below for schema)
    python examples/listwise/train.py --train_path data/train.jsonl

    # Or run with the built-in dummy data for a quick smoke test
    python examples/listwise/train.py --dummy
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from ir_torch.data import RankingCollator, RankingDataset
from ir_torch.nn.loss.listwise import ListNetLoss
from ir_torch.nn.loss.pointwise import PointwiseBCELoss

# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "jhu-clsp/ettin-decoder-17m"


class ListwiseRanker(L.LightningModule):
    """Wraps a transformer encoder into a listwise ranker."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        lr: float = 2e-5,
        loss: str = "listnet",
        label_max: float = 1.0,
        true_token: str = "true",  # noqa: S107
        false_token: str = "false",  # noqa: S107
        lm_head_name: str = "lm_head",
    ):
        super().__init__()
        self.save_hyperparameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load as sequence classifier
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        config.pad_token_id = tokenizer.pad_token_id
        config.attn_implementation = "flash_attn_2"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,  # ignore_misaligned_sizes=True,
        )

        # Initialize classifier head from true/false token weights
        self._init_head_from_lm(model_name, tokenizer, true_token, false_token, lm_head_name)

        if loss == "bce":
            self.criterion = PointwiseBCELoss(label_max=label_max)
        else:
            self.criterion = ListNetLoss()

    def _init_head_from_lm(
        self,
        model_name: str,
        tokenizer,
        true_token: str,
        false_token: str,
        lm_head_name: str = "lm_head",
    ):
        causal_lm = AutoModelForCausalLM.from_pretrained(model_name)
        lm_head = getattr(causal_lm, lm_head_name, None)
        if lm_head is None:
            raise AttributeError(  # noqa: TRY003
                f"Causal LM has no module named '{lm_head_name}'. "
                f"Available: {[n for n, _ in causal_lm.named_children()]}"
            )
        lm_head_weights = lm_head.weight.data

        true_id = tokenizer.convert_tokens_to_ids(true_token)
        false_id = tokenizer.convert_tokens_to_ids(false_token)

        classifier_vector = lm_head_weights[true_id] - lm_head_weights[false_id]

        # Find the classifier head (named 'score', 'classifier', etc.)
        head = None
        for name in ("score", "classifier"):
            head = getattr(self.model, name, None)
            if head is not None:
                break
        if head is None:
            return

        with torch.no_grad():
            head.weight.copy_(classifier_vector.unsqueeze(0))
            if head.bias is not None:
                head.bias.zero_()

        del causal_lm

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _step(self, batch, stage: str):
        batch_size, num_items, seq_len = batch.input_ids.shape

        # Flatten (batch * items, seq_len) for the transformer
        input_ids = batch.input_ids.view(-1, seq_len)
        attention_mask = batch.attention_mask.view(-1, seq_len)

        logits = self(input_ids, attention_mask)  # (batch * items, 1)
        logits = logits.view(batch_size, num_items, 1)

        loss = self.criterion(logits, batch.labels, item_mask=batch.item_mask)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ──────────────────────────────────────────────────────────────────────────────
# Dummy data (for quick smoke-testing)
# ──────────────────────────────────────────────────────────────────────────────

DUMMY_EXAMPLES = [
    {
        "query": "what is information retrieval",
        "items": [
            {"label": 3, "text": "Information retrieval is finding relevant documents."},
            {"label": 1, "text": "The weather is sunny today."},
            {"label": 2, "text": "Search engines use IR techniques."},
        ],
    },
    {
        "query": "neural ranking models",
        "items": [
            {"label": 2, "text": "BERT can be used for passage re-ranking."},
            {"label": 0, "text": "Cooking recipes are fun."},
            {"label": 3, "text": "Neural rankers learn relevance from data."},
        ],
    },
    {
        "query": "learning to rank",
        "items": [
            {"label": 3, "text": "LambdaRank optimises NDCG directly."},
            {"label": 1, "text": "Regression predicts a continuous value."},
        ],
    },
    {
        "query": "transformer architecture",
        "items": [
            {"label": 2, "text": "Transformers use self-attention."},
            {"label": 3, "text": "The original transformer was introduced in Attention Is All You Need."},
            {"label": 0, "text": "Gardening tips for spring."},
            {"label": 1, "text": "RNNs were popular before transformers."},
        ],
    },
]


def make_dummy_data(directory: Path) -> Path:
    path = directory / "dummy_train.jsonl"
    # Repeat a few times to have enough batches for a demo
    with open(path, "w") as f:
        for _ in range(8):
            for ex in DUMMY_EXAMPLES:
                f.write(json.dumps(ex) + "\n")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a listwise ranker with ListNet loss")
    p.add_argument("--train_path", type=str, default=None, help="Path to training JSONL file or directory")
    p.add_argument("--val_path", type=str, default=None, help="Path to validation JSONL file or directory")
    p.add_argument("--dummy", action="store_true", help="Use built-in dummy data for a smoke test")
    p.add_argument("--model_name", type=str, default=MODEL_NAME)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--loss", type=str, default="listnet", choices=["listnet", "bce"], help="Loss function")
    p.add_argument("--label_max", type=float, default=1.0, help="Max relevance grade for BCE normalisation")
    p.add_argument("--true_token", type=str, default="true", help="Token for 'relevant'")
    p.add_argument("--false_token", type=str, default="false", help="Token for 'irrelevant'")
    p.add_argument(
        "--lm_head_name",
        type=str,
        default="lm_head",
        help="Name of the output embedding module (e.g. lm_head, decoder, embed_out)",
    )
    p.add_argument(
        "--template",
        type=str,
        default="Query: {query} Document: {text} Relevant:",
        help="Prompt template with {query} and/or {text} placeholders",
    )
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Data ──────────────────────────────────────────────────────────────
    tmp_dir = None
    if args.dummy or args.train_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        train_path = make_dummy_data(Path(tmp_dir.name))
        val_path = train_path  # reuse for demo
    else:
        train_path = args.train_path
        val_path = args.val_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = RankingCollator(tokenizer=tokenizer, max_length=args.max_length, template=args.template)

    train_ds = RankingDataset(train_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=12,
    )

    val_loader = None
    if val_path is not None:
        val_ds = RankingDataset(val_path)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=12,
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model = ListwiseRanker(
        model_name=args.model_name,
        lr=args.lr,
        loss=args.loss,
        label_max=args.label_max,
        true_token=args.true_token,
        false_token=args.false_token,
        lm_head_name=args.lm_head_name,
    ).train()

    # ── Training ──────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("medium")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision="bf16",
        log_every_n_steps=1,
        enable_checkpointing=False,
        default_root_dir="/tmp/lightning",  # noqa: S108
        limit_train_batches=100,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    if not args.dummy:
        model.model.save_pretrained("/tmp/listwise_ranker")  # noqa: S108
        tokenizer.save_pretrained("/tmp/listwise_ranker")  # noqa: S108

    print("Done!")


if __name__ == "__main__":
    main()
