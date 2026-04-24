from __future__ import annotations

from time import perf_counter

from privategate.types import Mode

from ..harness.schemas import GoldItem, RouterDecision


class AlwaysHE:
    """B2 — always strictest backend. Privacy upper bound."""

    name = "always_he"

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        return RouterDecision(
            item_id=item.id,
            mode=Mode.HE_SMPC,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale="baseline:always_he",
        )
