from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import torch

Label = Union[int, float, Sequence[Union[int, float]]]


@dataclass
class RankingItem:
    label: Label
    text: str | None = None
    features: Sequence[int | float] | None = None


@dataclass
class RankingExample:
    items: list[RankingItem]
    query: str | None = None


@dataclass
class RankingBatch:
    labels: torch.Tensor
    item_mask: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    features: torch.Tensor | None = None

    _FIELDS = ("input_ids", "attention_mask", "features", "labels", "item_mask")

    def __getitem__(self, key: str) -> torch.Tensor | None:
        return getattr(self, key)

    def keys(self) -> list[str]:
        return [k for k in self._FIELDS if getattr(self, k) is not None]

    def values(self) -> list[torch.Tensor]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, torch.Tensor]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to(self, device: str | torch.device) -> RankingBatch:
        return RankingBatch(**{k: v.to(device) for k, v in self.items()})

    def pin_memory(self) -> RankingBatch:
        return RankingBatch(**{k: v.pin_memory() for k, v in self.items()})
