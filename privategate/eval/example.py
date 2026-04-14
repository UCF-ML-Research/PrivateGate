"""Normalized record format used across every dataset loader.

Real datasets (TAB, WikiPII, MedQA) ship in their own native formats. The
loaders convert each native format into this shape so the metrics and
baselines never have to special-case a dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from privategate.types import Category, RiskLevel, Span


@dataclass
class EvalExample:
    id: str
    text: str
    spans: list[Span] = field(default_factory=list)
    task: str = "generative"  # "generative" | "qa"
    question: Optional[str] = None
    reference_answer: Optional[str] = None


def span_from_dict(text: str, raw: dict) -> Span:
    start = int(raw["start"])
    end = int(raw["end"])
    return Span(
        start=start,
        end=end,
        text=text[start:end],
        category=Category(raw["category"]),
        risk=RiskLevel(raw.get("risk", "high")),
        source=raw.get("source", "gold"),
    )
