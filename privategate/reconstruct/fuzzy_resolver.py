"""Fuzzy placeholder resolution.

The plan (§4.6) requires that paraphrased references to a placeholder be
resolvable. The full embedding-based resolver lives in M3 once the ML deps
are wired up; here we provide a deterministic lexical fallback that detects
unresolved placeholders and inserts a visible marker rather than dropping
them silently.
"""
from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r"\[(?:SLOT|IDENTIFIER|CREDENTIAL|MEDICAL|FINANCIAL|PERSONAL_CONTEXT)_[A-Z0-9]+\]")
UNRESOLVED_MARKER = "<unresolved-placeholder>"


def resolve_paraphrased(response: str, placeholder_map: dict[str, str]) -> str:
    """Mark any leftover placeholder tokens that were not resolved verbatim."""
    def _sub(match: re.Match) -> str:
        token = match.group(0)
        return placeholder_map.get(token, UNRESOLVED_MARKER)

    return PLACEHOLDER_RE.sub(_sub, response)
