import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from ir_torch.data import (
    IterableRankingDataset,
    RankingBatch,
    RankingCollator,
    RankingDataset,
    RankingExample,
    RankingItem,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_DATA = [
    {
        "query": "what is python",
        "items": [
            {"label": 3, "text": "Python is a programming language.", "features": [0.9, 0.1]},
            {"label": 1, "text": "Java is also a language.", "features": [0.3, 0.7]},
        ],
    },
    {
        "query": "deep learning",
        "items": [
            {"label": 2, "text": "Neural networks are powerful.", "features": [0.8, 0.2]},
            {"label": 0, "text": "Linear regression is simple.", "features": [0.1, 0.9]},
        ],
    },
]

FEATURE_ONLY_DATA = [
    {
        "items": [
            {"label": 1, "features": [0.5, 0.3, 0.2]},
            {"label": 0, "features": [0.1, 0.8, 0.1]},
        ],
    },
    {
        "items": [
            {"label": 2, "features": [0.7, 0.1, 0.2]},
            {"label": 1, "features": [0.4, 0.4, 0.2]},
        ],
    },
]

MULTI_LABEL_DATA = [
    {
        "items": [
            {"label": [1, 0, 3], "text": "doc a"},
            {"label": [0, 2, 1], "text": "doc b"},
        ],
    },
]

VARIABLE_LENGTH_DATA = [
    {
        "items": [
            {"label": 3, "features": [0.9, 0.1]},
            {"label": 1, "features": [0.3, 0.7]},
            {"label": 2, "features": [0.5, 0.5]},
        ],
    },
    {
        "items": [
            {"label": 2, "features": [0.8, 0.2]},
        ],
    },
]


@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in EXAMPLE_DATA) + "\n")
    return path


@pytest.fixture
def jsonl_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    d.mkdir()
    (d / "shard_00.jsonl").write_text(json.dumps(EXAMPLE_DATA[0]) + "\n")
    (d / "shard_01.json").write_text(json.dumps(EXAMPLE_DATA[1]) + "\n")
    return d


@pytest.fixture
def feature_jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "features.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in FEATURE_ONLY_DATA) + "\n")
    return path


@pytest.fixture
def multi_label_jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "multi_label.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in MULTI_LABEL_DATA) + "\n")
    return path


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestItem:
    def test_scalar_label(self):
        item = RankingItem(label=3, text="hello")
        assert item.label == 3
        assert item.text == "hello"
        assert item.features is None

    def test_list_label(self):
        item = RankingItem(label=[1, 0, 3])
        assert item.label == [1, 0, 3]

    def test_with_features(self):
        item = RankingItem(label=1, features=[0.5, 0.3])
        assert item.features == [0.5, 0.3]


class TestRankingExample:
    def test_with_query(self):
        ex = RankingExample(
            query="hello",
            items=[RankingItem(label=1, text="world")],
        )
        assert ex.query == "hello"
        assert len(ex.items) == 1

    def test_without_query(self):
        ex = RankingExample(items=[RankingItem(label=0)])
        assert ex.query is None


class TestRankingBatch:
    def test_dict_access(self):
        batch = RankingBatch(labels=torch.tensor([[[1.0]]]))
        assert torch.equal(batch["labels"], batch.labels)

    def test_keys_excludes_none(self):
        batch = RankingBatch(labels=torch.tensor([[[1.0]]]))
        assert batch.keys() == ["labels"]

    def test_keys_includes_present(self):
        batch = RankingBatch(
            labels=torch.tensor([[[1.0]]]),
            features=torch.tensor([[[0.5]]]),
        )
        assert set(batch.keys()) == {"labels", "features"}

    def test_items(self):
        labels = torch.tensor([[[1.0]]])
        batch = RankingBatch(labels=labels)
        items = batch.items()
        assert len(items) == 1
        assert items[0][0] == "labels"
        assert torch.equal(items[0][1], labels)

    def test_to(self):
        batch = RankingBatch(labels=torch.tensor([[[1.0]]]))
        moved = batch.to("cpu")
        assert torch.equal(moved.labels, batch.labels)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_pin_memory(self):
        batch = RankingBatch(labels=torch.tensor([[[1.0]]]))
        pinned = batch.pin_memory()
        assert torch.equal(pinned.labels, batch.labels)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestRankingDataset:
    def test_len(self, jsonl_path: Path):
        ds = RankingDataset(jsonl_path)
        assert len(ds) == 2

    def test_getitem(self, jsonl_path: Path):
        ds = RankingDataset(jsonl_path)
        ex = ds[0]
        assert isinstance(ex, RankingExample)
        assert ex.query == "what is python"
        assert len(ex.items) == 2
        assert ex.items[0].label == 3
        assert ex.items[0].text == "Python is a programming language."
        assert ex.items[0].features == [0.9, 0.1]

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "blanks.jsonl"
        path.write_text(json.dumps(EXAMPLE_DATA[0]) + "\n\n" + json.dumps(EXAMPLE_DATA[1]) + "\n\n")
        ds = RankingDataset(path)
        assert len(ds) == 2

    def test_from_directory(self, jsonl_dir: Path):
        ds = RankingDataset(jsonl_dir)
        assert len(ds) == 2
        assert ds[0].query == "what is python"
        assert ds[1].query == "deep learning"

    def test_single_json_file(self, tmp_path: Path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps(EXAMPLE_DATA[0]) + "\n")
        ds = RankingDataset(path)
        assert len(ds) == 1
        assert ds[0].query == "what is python"

    def test_empty_directory_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            RankingDataset(empty)


class TestRankingIterableDataset:
    def test_iteration(self, jsonl_path: Path):
        ds = IterableRankingDataset(jsonl_path)
        examples = list(ds)
        assert len(examples) == 2
        assert examples[0].query == "what is python"

    def test_re_iterable(self, jsonl_path: Path):
        ds = IterableRankingDataset(jsonl_path)
        assert len(list(ds)) == len(list(ds))

    def test_from_directory(self, jsonl_dir: Path):
        ds = IterableRankingDataset(jsonl_dir)
        examples = list(ds)
        assert len(examples) == 2
        assert examples[0].query == "what is python"
        assert examples[1].query == "deep learning"

    def test_empty_directory_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = IterableRankingDataset(empty)
        with pytest.raises(FileNotFoundError):
            list(ds)


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestRankingCollatorFeaturesOnly:
    def test_labels_shape(self, feature_jsonl_path: Path):
        ds = RankingDataset(feature_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.labels.shape == (2, 2, 1)

    def test_features_shape(self, feature_jsonl_path: Path):
        ds = RankingDataset(feature_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.features is not None
        assert batch.features.shape == (2, 2, 3)

    def test_no_text(self, feature_jsonl_path: Path):
        ds = RankingDataset(feature_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.input_ids is None
        assert batch.attention_mask is None

    def test_label_values(self, feature_jsonl_path: Path):
        ds = RankingDataset(feature_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.labels[0, 0, 0].item() == 1.0
        assert batch.labels[0, 1, 0].item() == 0.0


class TestRankingCollatorMultiLabel:
    def test_multi_label_shape(self, multi_label_jsonl_path: Path):
        ds = RankingDataset(multi_label_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.labels.shape == (1, 2, 3)


class TestRankingCollatorWithTokenizer:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def test_text_shapes(self, jsonl_path: Path, tokenizer):
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer, max_length=32)
        batch = collator(list(ds))

        assert batch.input_ids is not None
        assert batch.attention_mask is not None
        # batch=2, docs=2, tokens<=32
        assert batch.input_ids.shape[0] == 2
        assert batch.input_ids.shape[1] == 2
        assert batch.input_ids.shape[2] <= 32

    def test_left_padding(self, jsonl_path: Path, tokenizer):
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer, max_length=32)
        batch = collator(list(ds))

        # Left-padded means first tokens should be pad tokens where attention_mask is 0
        for b in range(batch.input_ids.shape[0]):
            for d in range(batch.input_ids.shape[1]):
                mask = batch.attention_mask[b, d]
                # If there's padding, it should be on the left
                pad_positions = (mask == 0).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    # All pad positions should be contiguous from the start
                    assert pad_positions[-1].item() == len(pad_positions) - 1

    def test_tokenizer_padding_side_restored(self, jsonl_path: Path, tokenizer):
        original_side = tokenizer.padding_side
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer)
        collator(list(ds))
        assert tokenizer.padding_side == original_side

    def test_labels_present(self, jsonl_path: Path, tokenizer):
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer)
        batch = collator(list(ds))
        assert batch.labels.shape == (2, 2, 1)

    def test_features_present(self, jsonl_path: Path, tokenizer):
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer)
        batch = collator(list(ds))
        assert batch.features is not None
        assert batch.features.shape == (2, 2, 2)


class TestRankingCollatorContentOnly:
    """Test collation when items have content but no query."""

    @pytest.fixture
    def content_only_path(self, tmp_path: Path) -> Path:
        data = [
            {
                "items": [
                    {"label": 1, "text": "Python overview"},
                    {"label": 0, "text": "Java overview"},
                ],
            },
        ]
        path = tmp_path / "content_only.jsonl"
        path.write_text(json.dumps(data[0]) + "\n")
        return path

    def test_content_only(self, content_only_path: Path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        ds = RankingDataset(content_only_path)
        collator = RankingCollator(tokenizer=tokenizer, max_length=16)
        batch = collator(list(ds))
        assert batch.input_ids is not None
        assert batch.input_ids.shape[:2] == (1, 2)


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_map_dataset_with_dataloader(self, jsonl_path: Path):
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator()
        loader = DataLoader(ds, batch_size=2, collate_fn=collator)
        batch = next(iter(loader))
        assert isinstance(batch, RankingBatch)
        assert batch.labels.shape == (2, 2, 1)

    def test_iterable_dataset_with_dataloader(self, jsonl_path: Path):
        ds = IterableRankingDataset(jsonl_path)
        collator = RankingCollator()
        loader = DataLoader(ds, batch_size=2, collate_fn=collator)
        batch = next(iter(loader))
        assert isinstance(batch, RankingBatch)
        assert batch.labels.shape == (2, 2, 1)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestRankingCollatorWithTemplate:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def test_template_produces_input_ids(self, jsonl_path: Path, tokenizer):
        template = "Query: {query} Document: {text} Relevant:"
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer, max_length=64, template=template)
        batch = collator(list(ds))
        assert batch.input_ids is not None
        assert batch.attention_mask is not None
        assert batch.input_ids.shape[0] == 2
        assert batch.input_ids.shape[1] == 2
        assert batch.input_ids.shape[2] <= 64

    def test_template_content_present(self, jsonl_path: Path, tokenizer):
        """Verify the template text actually gets tokenized (not just query/content pairs)."""
        template = "Query: {query} Document: {text} Relevant:"
        ds = RankingDataset(jsonl_path)
        collator_tpl = RankingCollator(tokenizer=tokenizer, max_length=64, template=template)
        collator_plain = RankingCollator(tokenizer=tokenizer, max_length=64)
        batch_tpl = collator_tpl(list(ds))
        batch_plain = collator_plain(list(ds))
        # Template adds extra words so sequences should differ
        assert not torch.equal(batch_tpl.input_ids, batch_plain.input_ids)

    def test_template_query_only(self, tokenizer):
        """Template with only {query} placeholder."""
        template = "Is this relevant? {query}"
        examples = [
            RankingExample(query="test query", items=[RankingItem(label=1), RankingItem(label=0)]),
        ]
        collator = RankingCollator(tokenizer=tokenizer, max_length=32, template=template)
        batch = collator(examples)
        assert batch.input_ids is not None
        assert batch.input_ids.shape[:2] == (1, 2)

    def test_template_content_only(self, tokenizer):
        """Template with only {text} placeholder."""
        template = "Passage: {text}"
        examples = [
            RankingExample(
                items=[
                    RankingItem(label=1, text="hello world"),
                    RankingItem(label=0, text="foo bar"),
                ]
            ),
        ]
        collator = RankingCollator(tokenizer=tokenizer, max_length=32, template=template)
        batch = collator(examples)
        assert batch.input_ids is not None
        assert batch.input_ids.shape[:2] == (1, 2)

    def test_template_no_tokenizer_returns_no_text(self):
        """Template without a tokenizer should produce no text fields."""
        examples = [
            RankingExample(query="q", items=[RankingItem(label=1, text="c")]),
        ]
        collator = RankingCollator(template="Q: {query} D: {text}")
        batch = collator(examples)
        assert batch.input_ids is None

    def test_template_left_padding(self, jsonl_path: Path, tokenizer):
        template = "Query: {query} Document: {text} Relevant:"
        ds = RankingDataset(jsonl_path)
        collator = RankingCollator(tokenizer=tokenizer, max_length=64, template=template)
        batch = collator(list(ds))
        for b in range(batch.input_ids.shape[0]):
            for d in range(batch.input_ids.shape[1]):
                mask = batch.attention_mask[b, d]
                pad_positions = (mask == 0).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    assert pad_positions[-1].item() == len(pad_positions) - 1


# ---------------------------------------------------------------------------
# Item mask / variable-length tests
# ---------------------------------------------------------------------------


class TestItemMask:
    @pytest.fixture
    def variable_jsonl_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "variable.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in VARIABLE_LENGTH_DATA) + "\n")
        return path

    def test_item_mask_none_when_uniform(self, feature_jsonl_path: Path):
        ds = RankingDataset(feature_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.item_mask is None

    def test_item_mask_shape(self, variable_jsonl_path: Path):
        ds = RankingDataset(variable_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        # padded to max items = 3
        assert batch.item_mask is not None
        assert batch.item_mask.shape == (2, 3)

    def test_item_mask_values(self, variable_jsonl_path: Path):
        ds = RankingDataset(variable_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        # first example: 3 items -> all True
        assert batch.item_mask[0].tolist() == [True, True, True]
        # second example: 1 item -> [True, False, False]
        assert batch.item_mask[1].tolist() == [True, False, False]

    def test_labels_padded(self, variable_jsonl_path: Path):
        ds = RankingDataset(variable_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.labels.shape == (2, 3, 1)
        # padded label slots should be 0
        assert batch.labels[1, 1, 0].item() == 0.0
        assert batch.labels[1, 2, 0].item() == 0.0

    def test_features_padded(self, variable_jsonl_path: Path):
        ds = RankingDataset(variable_jsonl_path)
        collator = RankingCollator()
        batch = collator(list(ds))
        assert batch.features is not None
        assert batch.features.shape == (2, 3, 2)
        # padded feature slots should be 0
        assert batch.features[1, 1].tolist() == [0.0, 0.0]
        assert batch.features[1, 2].tolist() == [0.0, 0.0]
