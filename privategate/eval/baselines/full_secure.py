"""Baseline #4 — send every query to the secure backend.

Upper bound on privacy: nothing reaches the standard backend, so the
outbound payload to an *untrusted* server is empty by construction. We
report `outbound_text = ""` and `routing_path = "secure"`.
"""
from __future__ import annotations

from privategate.backends.base import Backend
from privategate.backends.mock_secure import MockSecureBackend
from privategate.eval.baselines.base import Baseline, BaselineRun, _timed
from privategate.eval.example import EvalExample


class FullSecureBaseline(Baseline):
    name = "full_secure"

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend or MockSecureBackend()

    def run_one(self, example: EvalExample) -> BaselineRun:
        answer, ms = _timed(lambda: self._backend.complete(example.text))
        return BaselineRun(
            name=self.name,
            outbound_text="",
            answer=answer,
            routing_path="secure",
            latency_ms=ms,
        )
