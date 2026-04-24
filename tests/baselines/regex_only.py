"""Regex-only router (B3).

Deterministic PII patterns. Any match → LDP; else plaintext. No HE path — plain
regex cannot distinguish `strictly-must-not-release` from ordinary PII.
"""
from __future__ import annotations

import re
from time import perf_counter

from privategate.types import Mode

from ..harness.schemas import GoldItem, RouterDecision

_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|xoxb|ghp|gho)[-_][A-Za-z0-9]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
}


class RegexOnly:
    """B3 — regex PII detection only."""

    name = "regex_only"

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        hits: list[str] = [k for k, pat in _PATTERNS.items() if pat.search(item.prompt)]
        mode = Mode.PERTURB if hits else Mode.PLAINTEXT
        return RouterDecision(
            item_id=item.id,
            mode=mode,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale=f"regex_hits={hits or 'none'}",
        )
