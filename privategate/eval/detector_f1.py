"""Span-level F1 for the detector.

Two modes:
- ``strict``: a predicted span counts as a true positive only if its
  (start, end, category) exactly matches a gold span.
- ``overlap``: a predicted span counts if it overlaps a gold span of the
  same category by at least one character.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from privategate.types import Span


@dataclass(frozen=True)
class F1Report:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _spans_overlap_same_cat(a: Span, b: Span) -> bool:
    if a.category is not b.category:
        return False
    return not (a.end <= b.start or b.end <= a.start)


def compute_f1(
    pred: Sequence[Span],
    gold: Sequence[Span],
    mode: str = "strict",
) -> F1Report:
    if mode not in {"strict", "overlap"}:
        raise ValueError(f"unknown mode: {mode}")

    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if mode == "strict":
                hit = (p.start == g.start and p.end == g.end and p.category is g.category)
            else:
                hit = _spans_overlap_same_cat(p, g)
            if hit:
                tp += 1
                matched_gold.add(i)
                break

    fp = len(pred) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return F1Report(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def aggregate(reports: Iterable[F1Report]) -> F1Report:
    tp = fp = fn = 0
    for r in reports:
        tp += r.tp
        fp += r.fp
        fn += r.fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return F1Report(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)
