from __future__ import annotations

from time import perf_counter

from privategate.types import Mode

from ..harness.schemas import GoldItem, RouterDecision


class AlwaysPlaintext:
    """B1 — never escalates. Privacy lower bound."""

    name = "always_plaintext"

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        return RouterDecision(
            item_id=item.id,
            mode=Mode.PLAINTEXT,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale="baseline:always_plaintext",
        )
