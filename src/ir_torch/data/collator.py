from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from .types import RankingBatch, RankingExample


class RankingCollator:
    """Collates :class:`RankingExample` instances into a :class:`RankingBatch`.

    Handles tokenization of query/item text (with left padding) and collation
    of labels and features into tensors. Examples with fewer items than the
    maximum are zero-padded, with ``item_mask`` indicating valid positions.

    Args:
        tokenizer: A HuggingFace tokenizer. Required if examples contain text.
        max_length: Maximum sequence length for tokenization.
        template: An optional prompt template string with ``{query}`` and/or
            ``{text}`` placeholders that will be populated per item before
            tokenization, e.g. ``"Query: {query} Document: {text} Relevant:"``.
            When provided, the formatted string is tokenized as a single
            sequence (no text pairs).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int | None = None,
        template: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def __call__(self, examples: list[RankingExample]) -> RankingBatch:
        examples, item_mask = self._pad_items(examples)
        return RankingBatch(
            labels=self._collate_labels(examples),
            item_mask=item_mask,
            features=self._collate_features(examples),
            **self._collate_text(examples),
        )

    @staticmethod
    def _pad_items(
        examples: list[RankingExample],
    ) -> tuple[list[RankingExample], torch.Tensor | None]:
        """Pad examples to the same number of items and return an item_mask."""
        from .types import RankingItem

        counts = [len(ex.items) for ex in examples]
        max_items = max(counts)

        if all(c == max_items for c in counts):
            return examples, None

        # Need a representative label shape (scalar vs list) from the first real item
        sample_label = examples[0].items[0].label
        pad_label: list | int | float
        pad_label = [0] * len(sample_label) if isinstance(sample_label, (list, tuple)) else 0

        padded: list[RankingExample] = []
        mask_rows: list[list[int]] = []
        for ex in examples:
            n = len(ex.items)
            pad_count = max_items - n
            items = list(ex.items)
            if pad_count > 0:
                # Determine feature width from this example or others
                feat_len = next(
                    (len(it.features) for it in ex.items if it.features is not None),
                    None,
                )
                pad_item = RankingItem(
                    label=pad_label,
                    features=[0] * feat_len if feat_len is not None else None,
                )
                items.extend([pad_item] * pad_count)
            padded.append(RankingExample(items=items, query=ex.query))
            mask_rows.append([1] * n + [0] * pad_count)

        return padded, torch.tensor(mask_rows, dtype=torch.bool)

    def _collate_labels(self, examples: list[RankingExample]) -> torch.Tensor:
        batch = []
        for ex in examples:
            item_labels = []
            for item in ex.items:
                label = list(item.label) if isinstance(item.label, (list, tuple)) else [item.label]
                item_labels.append(label)
            batch.append(item_labels)
        return torch.tensor(batch, dtype=torch.float)

    def _collate_features(self, examples: list[RankingExample]) -> torch.Tensor | None:
        if not any(item.features is not None for ex in examples for item in ex.items):
            return None
        batch = []
        for ex in examples:
            batch.append([list(item.features) if item.features is not None else [] for item in ex.items])
        return torch.tensor(batch, dtype=torch.float)

    def _collate_text(self, examples: list[RankingExample]) -> dict:
        if self.tokenizer is None:
            return {}

        batch_size = len(examples)
        num_items = len(examples[0].items)

        if self.template is not None:
            texts = self._apply_template(examples)
        else:
            texts = self._build_text_pairs(examples)
            if texts is None:
                return {}

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length

            if isinstance(texts, tuple):
                encoded = self.tokenizer(texts[0], texts[1], **kwargs)
            else:
                encoded = self.tokenizer(texts, **kwargs)
        finally:
            self.tokenizer.padding_side = orig_padding_side

        seq_len = encoded["input_ids"].shape[1]
        return {
            "input_ids": encoded["input_ids"].view(batch_size, num_items, seq_len),
            "attention_mask": encoded["attention_mask"].view(batch_size, num_items, seq_len),
        }

    def _apply_template(self, examples: list[RankingExample]) -> list[str]:
        """Format each item through the prompt template."""
        texts: list[str] = []
        for ex in examples:
            for item in ex.items:
                texts.append(
                    self.template.format(
                        query=ex.query or "",
                        text=item.text or "",
                    )
                )
        return texts

    def _build_text_pairs(self, examples: list[RankingExample]) -> list[str] | tuple[list[str], list[str]] | None:
        """Build text / text-pair lists for default (non-template) tokenization."""
        has_query = any(ex.query is not None for ex in examples)
        has_content = any(item.text is not None for ex in examples for item in ex.items)

        if not has_query and not has_content:
            return None

        texts: list[str] = []
        text_pairs: list[str] | None = [] if (has_query and has_content) else None

        for ex in examples:
            for item in ex.items:
                if text_pairs is not None:
                    texts.append(ex.query or "")
                    text_pairs.append(item.text or "")
                elif has_content:
                    texts.append(item.text or "")
                else:
                    texts.append(ex.query or "")

        if text_pairs is not None:
            return (texts, text_pairs)
        return texts
