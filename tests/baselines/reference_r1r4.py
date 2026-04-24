"""Oracle R1–R4 router (B6).

Feeds the gold labels straight into the reference policy engine. Not
deployable (real-time gold labels are not available at inference time), but
gives the *upper bound* on routing accuracy assuming perfect classification.
The gap between PrivateGate and this oracle is purely classifier error.
"""
from __future__ import annotations

from time import perf_counter

from privategate.policy.engine import PolicyEngine
from privategate.policy.reference_rules import REFERENCE_RULES, REFERENCE_STRICTNESS
from privategate.types import Context, Request, Signal

from ..harness.schemas import GoldItem, RouterDecision


class OracleR1R4:
    """B6 — perfect-classifier + R1-R4 reference policy; routing-accuracy upper bound."""

    name = "oracle_r1r4"

    def __init__(self) -> None:
        self.engine = PolicyEngine(REFERENCE_RULES, REFERENCE_STRICTNESS)

    def _signals(self, item: GoldItem) -> list[Signal]:
        signals: list[Signal] = []
        for cat, present in item.categories.items():
            if not present:
                continue
            severity = int(item.severity.get(cat, 0))
            signals.append(
                Signal(
                    type=f"category.{cat}",
                    score=1.0,
                    source="oracle@gold",
                    evidence={"severity": severity},
                )
            )
        return signals

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        decision = self.engine.evaluate(
            Request(payload=item.prompt, context=Context()),
            self._signals(item),
        )
        return RouterDecision(
            item_id=item.id,
            mode=decision.mode,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale=decision.rationale,
            matched_rule=decision.matched_rule,
            confidence=decision.confidence,
        )
