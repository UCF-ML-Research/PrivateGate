"""Metrics for the PrivateGate v1 evaluation.

Strictness order (least → most restrictive):
    PLAINTEXT < PERTURB (LDP) < HE_SMPC (HE) < ABSTAIN

- False-downgrade: router picked a mode *less* restrictive than the gold mode.
  This is the primary safety metric.
- Over-escalation: router picked a mode *more* restrictive than the gold mode
  (ABSTAIN excluded — counted separately).
"""
from __future__ import annotations

from privategate.types import Mode

from .schemas import EvalResult, GoldItem, RouterDecision

# v1 strictness ordering. REDACT/TEE are not active in v1 and are intentionally
# omitted; adding them later only requires inserting them at the right rank.
STRICTNESS: dict[Mode, int] = {
    Mode.PLAINTEXT: 0,
    Mode.PERTURB: 1,
    Mode.HE_SMPC: 2,
    Mode.ABSTAIN: 3,
}


def is_downgrade(predicted: Mode, gold: Mode) -> bool:
    return STRICTNESS[predicted] < STRICTNESS[gold]


def is_over_escalation(predicted: Mode, gold: Mode) -> bool:
    # ABSTAIN is a safety valve, not an over-escalation.
    if predicted == Mode.ABSTAIN or gold == Mode.ABSTAIN:
        return False
    return STRICTNESS[predicted] > STRICTNESS[gold]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def aggregate(
    router: str,
    items: list[GoldItem],
    decisions: dict[str, RouterDecision],
) -> EvalResult:
    """Compute EvalResult over the subset of items for which we have a decision."""
    n = 0
    n_downgrade = 0
    n_over = 0
    n_abstain = 0
    per_cat: dict[str, list[int]] = {}
    latencies: list[float] = []
    confusion: dict[tuple[str, str], int] = {}

    for item in items:
        d = decisions.get(item.id)
        if d is None:
            continue
        n += 1
        latencies.append(d.latency_ms)

        if d.mode == Mode.ABSTAIN:
            n_abstain += 1

        downgraded = is_downgrade(d.mode, item.gold_mode)
        if downgraded:
            n_downgrade += 1

        # Per-class FDR: counted only for categories marked present.
        for cat, present in item.categories.items():
            if not present:
                continue
            per_cat.setdefault(cat, []).append(1 if downgraded else 0)

        if is_over_escalation(d.mode, item.gold_mode):
            n_over += 1

        key = (item.gold_mode.value, d.mode.value)
        confusion[key] = confusion.get(key, 0) + 1

    fdr_per_class = {
        cat: (sum(xs) / len(xs)) if xs else 0.0 for cat, xs in per_cat.items()
    }

    return EvalResult(
        router=router,
        n=n,
        fdr_overall=(n_downgrade / n) if n else 0.0,
        fdr_per_class=fdr_per_class,
        oer=(n_over / n) if n else 0.0,
        abstain_rate=(n_abstain / n) if n else 0.0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        confusion=confusion,
    )
