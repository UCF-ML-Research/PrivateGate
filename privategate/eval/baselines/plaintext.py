"""Baseline #1 — send the raw query to the standard backend.

Upper bound on utility, zero privacy. Reports `routing_path = standard`.
"""
from __future__ import annotations

from privategate.backends.base import Backend
from privategate.backends.mock_standard import MockStandardBackend
from privategate.eval.baselines.base import Baseline, BaselineRun, _timed
from privategate.eval.example import EvalExample


class PlaintextBaseline(Baseline):
    name = "plaintext"

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or MockStandardBackend()

    def run_one(self, example: EvalExample) -> BaselineRun:
        answer, ms = _timed(lambda: self._backend.complete(example.text))
        return BaselineRun(
            name=self.name,
            outbound_text=example.text,
            answer=answer,
            routing_path="standard",
            latency_ms=ms,
        )
