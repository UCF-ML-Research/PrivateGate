"""JSONL schema + loader for privategate-gold.

Uses user-facing aliases `{plaintext, ldp, he, abstain}` in the JSONL file and
maps them to canonical `Mode` values. See `harness/schemas.py` for the
alias tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..harness.schemas import JSONL_TO_MODE, GoldItem

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "none",
    "pii",
    "phi",
    "pci",
    "secret",
    "ip_confidential",
    "regulated_eu",
    "regulated_us",
    "injection",
)

_REQUIRED_FIELDS = ("id", "prompt", "source", "categories", "gold_mode")
# Canonical set plus the raw-corpus names emitted by download.py and
# prepare_pilot.py — preserves provenance through to the gold JSONL.
_ALLOWED_SOURCES = {
    "sharegpt", "wildchat", "synthetic", "adversarial", "a2a",
    "oasst1", "ai4privacy", "jailbreakbench",
}


def validate_item(obj: dict[str, Any]) -> GoldItem:
    """Validate a single JSONL record and return a typed GoldItem.

    Raises ValueError on any schema violation.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in obj]
    if missing:
        raise ValueError(f"missing fields: {missing}")

    if obj["source"] not in _ALLOWED_SOURCES:
        raise ValueError(f"invalid source {obj['source']!r}; allowed: {_ALLOWED_SOURCES}")

    cats = obj["categories"]
    if not isinstance(cats, dict):
        raise ValueError("categories must be a dict[str, bool]")
    for c in REQUIRED_CATEGORIES:
        if c not in cats:
            raise ValueError(f"categories missing key: {c}")
        if not isinstance(cats[c], bool):
            raise ValueError(f"categories[{c}] must be bool, got {type(cats[c]).__name__}")

    # Semantic checks: `none` iff all other categories false.
    any_other = any(cats[c] for c in REQUIRED_CATEGORIES if c != "none")
    if cats["none"] == any_other:
        raise ValueError(
            "categories.none must be true iff all other categories are false "
            f"(got none={cats['none']}, any_other={any_other})"
        )

    severity = obj.get("severity", {})
    if not isinstance(severity, dict):
        raise ValueError("severity must be a dict[str, int]")
    for k, v in severity.items():
        if k not in REQUIRED_CATEGORIES:
            raise ValueError(f"severity key {k!r} not in REQUIRED_CATEGORIES")
        if not isinstance(v, int) or not 0 <= v <= 3:
            raise ValueError(f"severity[{k}] must be int in 0..3, got {v!r}")

    mode_alias = obj["gold_mode"]
    if mode_alias not in JSONL_TO_MODE:
        raise ValueError(
            f"invalid gold_mode {mode_alias!r}; allowed: {sorted(JSONL_TO_MODE)}"
        )

    spans = obj.get("spans", [])
    if not isinstance(spans, list):
        raise ValueError("spans must be a list")
    for s in spans:
        if not {"start", "end", "type"}.issubset(s):
            raise ValueError(f"span missing required keys: {s}")

    return GoldItem(
        id=str(obj["id"]),
        prompt=str(obj["prompt"]),
        source=str(obj["source"]),
        categories=cats,
        severity=severity,
        spans=spans,
        gold_mode=JSONL_TO_MODE[mode_alias],
        notes=str(obj.get("notes", "")),
        annotators=tuple(obj.get("annotators", ())),
        adjudicated=bool(obj.get("adjudicated", False)),
    )


def load_jsonl(path: str | Path) -> list[GoldItem]:
    """Load and validate a JSONL file; raises on any invalid line."""
    items: list[GoldItem] = []
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(validate_item(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as e:
                raise ValueError(f"{p}:{i} — {e}") from e
    return items
