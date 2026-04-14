from __future__ import annotations

from pathlib import Path
from typing import Optional

from privategate.eval.datasets._jsonl import load_jsonl
from privategate.eval.example import EvalExample

_FIXTURE = Path(__file__).parent / "fixtures" / "wikipii_tiny.jsonl"


def load_wikipii(path: Optional[Path | str] = None) -> list[EvalExample]:
    return load_jsonl(path or _FIXTURE)
