"""Recovery-rate scorer for adversarial evaluation.

A gold span is *recovered* if any of the attacker's predictions contains
its plaintext (case-insensitive substring match). We deliberately choose
substring rather than exact match: the threat model says the attacker
wins as soon as the secret leaves the client, even if it is wrapped in
extra context.

`recovery_rate` reports both the overall rate and a per-category
breakdown so the report can call out (e.g.) credential recovery in
isolation from softer personal-context recovery.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from privategate.types import Category, Span


@dataclass
class RecoveryReport:
    overall_rate: float
    n_gold: int
    n_recovered: int
    by_category: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_rate": round(self.overall_rate, 4),
            "n_gold": self.n_gold,
            "n_recovered": self.n_recovered,
            "by_category": {
                k: {"rate": round(v["rate"], 4), "n_gold": v["n_gold"], "n_recovered": v["n_recovered"]}
                for k, v in self.by_category.items()
            },
        }


def _normalized(text: str) -> str:
    return text.lower().strip()


def _is_recovered(span: Span, predictions: Sequence[str]) -> bool:
    needle = _normalized(span.text)
    if not needle:
        return False
    return any(needle in _normalized(p) for p in predictions)


def recovery_rate(
    predictions: Sequence[str],
    gold_spans: Sequence[Span],
    categories: Optional[Iterable[Category]] = None,
) -> RecoveryReport:
    if categories is not None:
        wanted = set(categories)
        gold_spans = [s for s in gold_spans if s.category in wanted]

    if not gold_spans:
        return RecoveryReport(overall_rate=0.0, n_gold=0, n_recovered=0, by_category={})

    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_recovered: dict[str, int] = defaultdict(int)
    n_recovered = 0
    for s in gold_spans:
        cat = s.category.value
        per_cat_total[cat] += 1
        if _is_recovered(s, predictions):
            n_recovered += 1
            per_cat_recovered[cat] += 1

    by_cat: dict[str, dict] = {}
    for cat, total in per_cat_total.items():
        rec = per_cat_recovered[cat]
        by_cat[cat] = {
            "rate": rec / total if total else 0.0,
            "n_gold": total,
            "n_recovered": rec,
        }

    return RecoveryReport(
        overall_rate=n_recovered / len(gold_spans),
        n_gold=len(gold_spans),
        n_recovered=n_recovered,
        by_category=by_cat,
    )
