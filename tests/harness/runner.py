"""Run a router over a dataset of GoldItems and aggregate metrics."""
from __future__ import annotations

from .metrics import aggregate
from .router import BaselineRouter
from .schemas import EvalResult, GoldItem


def run_eval(router: BaselineRouter, items: list[GoldItem]) -> EvalResult:
    decisions = {item.id: router.route(item) for item in items}
    return aggregate(router.name, items, decisions)
