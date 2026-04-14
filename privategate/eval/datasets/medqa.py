"""MedQA / PubMedQA subset loader.

Real loader will inject synthetic patient identifiers into reasoning
questions (plan §6.1). The fixture shipped here is hand-built so the
harness has at least one medical example to run on without network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from privategate.eval.datasets._jsonl import load_jsonl
from privategate.eval.example import EvalExample

_FIXTURE = Path(__file__).parent / "fixtures" / "medqa_tiny.jsonl"


def load_medqa_subset(path: Optional[Path | str] = None) -> list[EvalExample]:
    return load_jsonl(path or _FIXTURE)
