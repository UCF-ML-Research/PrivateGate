"""Privacy metrics.

These are computed against the **outbound payload** — i.e. the string
that actually leaves the trusted client. Anything in the placeholder map
is by definition still on the client and does not count as exposure.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from privategate.types import Category, Span


_DEFAULT_WEIGHTS: Mapping[Category, float] = {
    Category.CREDENTIAL: 4.0,
    Category.MEDICAL: 3.0,
    Category.FINANCIAL: 3.0,
    Category.IDENTIFIER: 2.0,
    Category.PERSONAL_CONTEXT: 1.0,
}


def span_exposure_rate(transformed: str, gold_spans: Sequence[Span]) -> float:
    """Fraction of gold spans whose verbatim text appears in `transformed`."""
    if not gold_spans:
        return 0.0
    leaked = sum(1 for s in gold_spans if s.text and s.text in transformed)
    return leaked / len(gold_spans)


def category_weighted_leakage(
    transformed: str,
    gold_spans: Sequence[Span],
    weights: Mapping[Category, float] = _DEFAULT_WEIGHTS,
) -> float:
    """Weighted leakage score in [0, 1]: sum(weights of leaked) / sum(weights of all)."""
    if not gold_spans:
        return 0.0
    total = 0.0
    leaked = 0.0
    for s in gold_spans:
        w = weights.get(s.category, 1.0)
        total += w
        if s.text and s.text in transformed:
            leaked += w
    if total == 0.0:
        return 0.0
    return leaked / total
