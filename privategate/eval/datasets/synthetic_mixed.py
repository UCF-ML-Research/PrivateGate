"""SyntheticMixed loader.

The seed file lives next to this module so it ships with the package and
can be exercised by both the unit tests and the main eval runner. Plan
§6.1 calls for ~500 hand-written queries; the seed is small enough to
review in a PR and we'll grow it in M7.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from privategate.eval.datasets._jsonl import load_jsonl
from privategate.eval.example import EvalExample

_SEED = Path(__file__).parent / "synthetic_mixed_seed.jsonl"


def load_synthetic_mixed(path: Optional[Path | str] = None) -> list[EvalExample]:
    return load_jsonl(path or _SEED)
