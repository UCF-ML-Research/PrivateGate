"""Baseline-router protocol used by all B1–B5 implementations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import GoldItem, RouterDecision


@runtime_checkable
class BaselineRouter(Protocol):
    """A router takes a gold item and produces a mode decision.

    Implementations must record their own decision latency in
    `RouterDecision.latency_ms` (in milliseconds, wall-clock).
    """

    name: str

    def route(self, item: GoldItem) -> RouterDecision: ...
