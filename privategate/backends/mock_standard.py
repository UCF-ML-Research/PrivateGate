"""Deterministic standard-backend mock for CI.

It echoes the prompt back inside an envelope. Tests rely on this being a
pure function so that pipeline behavior is reproducible.
"""
from __future__ import annotations

from privategate.backends.base import Backend


class MockStandardBackend(Backend):
    name = "mock-standard"

    def complete(self, prompt: str) -> str:
        return f"[STANDARD] received: {prompt}"
