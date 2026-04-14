"""Risk scoring with optional context bumps."""
from __future__ import annotations

from privategate.types import Category, RiskLevel, Span


_BASE_RISK: dict[Category, RiskLevel] = {
    Category.CREDENTIAL: RiskLevel.CRITICAL,
    Category.IDENTIFIER: RiskLevel.HIGH,
    Category.MEDICAL: RiskLevel.HIGH,
    Category.FINANCIAL: RiskLevel.HIGH,
    Category.PERSONAL_CONTEXT: RiskLevel.MEDIUM,
}

_RANK = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def _bump(r: RiskLevel, steps: int = 1) -> RiskLevel:
    idx = min(len(_RANK) - 1, max(0, _RANK.index(r) + steps))
    return _RANK[idx]


def score_risk(span: Span, context: str = "") -> RiskLevel:
    base = _BASE_RISK.get(span.category, span.risk)
    risk = max(base, span.risk, key=lambda r: r.rank)

    # context bump: medical term colocated with a name in the same sentence
    if span.category is Category.MEDICAL and context:
        snippet = context.lower()
        if any(p in snippet for p in (" mr.", " mrs.", " ms.", " dr.")):
            risk = _bump(risk)

    return risk
