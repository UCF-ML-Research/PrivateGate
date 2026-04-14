"""Shared JSONL loader for normalized dataset files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from privategate.eval.example import EvalExample, span_from_dict


def load_jsonl(path: Path | str) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} — invalid JSON: {exc}") from exc
            text = raw["text"]
            spans = [span_from_dict(text, s) for s in raw.get("spans", [])]
            examples.append(
                EvalExample(
                    id=str(raw.get("id", f"{Path(path).stem}-{line_no}")),
                    text=text,
                    spans=spans,
                    task=raw.get("task", "generative"),
                    question=raw.get("question"),
                    reference_answer=raw.get("reference_answer"),
                )
            )
    return examples
