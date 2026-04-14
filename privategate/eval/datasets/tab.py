"""TAB (Text Anonymization Benchmark) loader.

We don't bundle the real TAB corpus (license + size). The loader expects
a normalized JSONL file produced by `scripts/prepare_tab.py` (not yet
written — see plan §M5). A tiny fixture is shipped under `fixtures/`
so the harness can run hermetically in CI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from privategate.eval.datasets._jsonl import load_jsonl
from privategate.eval.example import EvalExample

_FIXTURE = Path(__file__).parent / "fixtures" / "tab_tiny.jsonl"


def load_tab(path: Optional[Path | str] = None) -> list[EvalExample]:
    return load_jsonl(path or _FIXTURE)
