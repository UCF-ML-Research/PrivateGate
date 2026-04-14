"""Per-query placeholder map.

IDs are random per query so a server cannot correlate placeholders across
sessions (plan §4.3, §8 — placeholder correlation risk).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field


@dataclass
class PlaceholderMap:
    _entries: dict[str, str] = field(default_factory=dict)
    _id_prefix: str = field(default_factory=lambda: secrets.token_hex(2).upper())
    _next: int = 0

    def add(self, label: str, original: str) -> str:
        token = f"[{label}_{self._id_prefix}{self._next}]"
        self._next += 1
        self._entries[token] = original
        return token

    def as_dict(self) -> dict[str, str]:
        return dict(self._entries)

    def __contains__(self, token: str) -> bool:
        return token in self._entries

    def __len__(self) -> int:
        return len(self._entries)
