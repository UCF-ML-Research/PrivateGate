"""Thin, torch-free adapter from privategate-gold JSONL to label vectors.

A torch-aware `Dataset` wrapper can be added in `train.py` once the trainer
is implemented; keeping this module torch-free means inspection / schema
verification runs without GPU stack installed.
"""
from __future__ import annotations

from pathlib import Path

from ..data.schema import load_jsonl
from ..harness.schemas import GoldItem


def load_split(path: str | Path) -> list[GoldItem]:
    return load_jsonl(path)


def to_multilabel(item: GoldItem, categories: list[str]) -> list[int]:
    """Return a 0/1 label vector aligned to `categories`."""
    return [int(bool(item.categories.get(c, False))) for c in categories]


def class_frequencies(items: list[GoldItem], categories: list[str]) -> dict[str, int]:
    counts = {c: 0 for c in categories}
    for item in items:
        for c in categories:
            if item.categories.get(c, False):
                counts[c] += 1
    return counts
