from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset

from .types import RankingExample, RankingItem


def _parse_example(data: dict) -> RankingExample:
    return RankingExample(
        query=data.get("query"),
        items=[
            RankingItem(
                label=item["label"],
                text=item.get("text", item.get("content")),
                features=item.get("features"),
            )
            for item in data.get("items", data.get("documents", []))
        ],
    )


_GLOB_PATTERNS = ("*.jsonl", "*.json")


def _resolve_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(f for pattern in _GLOB_PATTERNS for f in path.glob(pattern))
        if not files:
            msg = f"No .jsonl or .json files found in directory: {path}"
            raise FileNotFoundError(msg)
        return files
    return [path]


def _iter_lines(files: list[Path]) -> Iterator[str]:
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


class _RankingDatasetBase:
    @staticmethod
    def _parse_line(line: str) -> RankingExample:
        return _parse_example(json.loads(line))

    @staticmethod
    def _split_example(example: RankingExample, max_items: int) -> list[RankingExample]:
        """Split an example into sub-queries of at most *max_items* items."""
        items = example.items
        if len(items) <= max_items:
            return [example]
        return [
            RankingExample(query=example.query, items=items[i : i + max_items]) for i in range(0, len(items), max_items)
        ]


class RankingDataset(_RankingDatasetBase, Dataset):
    """Map-style dataset that loads ranking examples from a JSONL file or directory into memory.

    Args:
        path: Path to a JSONL/JSON file or directory of such files.
        max_items: If set, queries with more items are split into sub-queries
            of at most this many items.
    """

    def __init__(self, path: str | Path, max_items: int | None = None):
        self.examples: list[RankingExample] = []
        for line in _iter_lines(_resolve_files(path)):
            ex = self._parse_line(line)
            if max_items is not None:
                self.examples.extend(self._split_example(ex, max_items))
            else:
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RankingExample:
        return self.examples[index]


class IterableRankingDataset(_RankingDatasetBase, IterableDataset):
    """Streaming iterable dataset that lazily reads ranking examples from JSONL files.

    Args:
        path: Path to a JSONL/JSON file or directory of such files.
        max_items: If set, queries with more items are split into sub-queries
            of at most this many items.
    """

    def __init__(self, path: str | Path, max_items: int | None = None):
        super().__init__()
        self.path = path
        self.max_items = max_items

    def __iter__(self) -> Iterator[RankingExample]:
        for line in _iter_lines(_resolve_files(self.path)):
            ex = self._parse_line(line)
            if self.max_items is not None:
                yield from self._split_example(ex, self.max_items)
            else:
                yield ex
