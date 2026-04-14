"""Merge spans from multiple detectors.

Policy: rules are deterministic, so on overlap a rule span beats an ML
span. Among spans of the same source, the higher-risk span wins; ties are
broken by length (longer wins). The output is sorted and non-overlapping.
"""
from __future__ import annotations

from typing import Iterable

from privategate.types import Span


def _priority(span: Span) -> tuple[int, int, int]:
    source_rank = {"rule": 2, "gazetteer": 1, "ml": 0}.get(span.source, 0)
    return (source_rank, span.risk.rank, span.end - span.start)


def merge_spans(rule_spans: Iterable[Span], ml_spans: Iterable[Span]) -> list[Span]:
    all_spans = sorted(
        list(rule_spans) + list(ml_spans),
        key=lambda s: (s.start, -(s.end - s.start)),
    )
    out: list[Span] = []
    for span in all_spans:
        if not out:
            out.append(span)
            continue
        last = out[-1]
        if span.overlaps(last):
            keep = max((span, last), key=_priority)
            out[-1] = keep
        else:
            out.append(span)
    return out
