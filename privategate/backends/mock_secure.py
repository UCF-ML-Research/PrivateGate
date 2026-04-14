"""Deterministic secure-backend mock.

In the v1 plan this stands in for a locally-hosted small model running on
the trusted client. Tests use it to validate routing and reconstruction
without needing a real model.
"""
from __future__ import annotations

from privategate.backends.base import Backend


class MockSecureBackend(Backend):
    name = "mock-secure"

    def complete(self, prompt: str) -> str:
        return f"[SECURE] handled locally: {prompt}"
