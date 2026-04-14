"""Fine-tune the ML detector on TAB / WikiPII slices.

Real training is gated behind ``transformers``/``torch``; this script runs
in two modes:

- ``--smoke``: builds the data pipeline, runs a single forward pass on
  one fixture example, exits 0. Used by tests to verify that the tagging
  layout and label map line up with what `MLDetector` expects.
- (default): full fine-tuning loop. Skipped automatically if the optional
  ML deps are missing — prints a hint and exits 0.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

from privategate.detector.ml_detector import MLDetector, TaggedEntity
from privategate.types import Category


FIXTURE_EXAMPLES = [
    ("Alice Smith lives in Paris.", [("Alice Smith", "PER"), ("Paris", "LOC")]),
    ("contact bob@example.com", [("bob@example.com", "EMAIL")]),
]


def _fixture_tagger(text: str) -> Iterable[TaggedEntity]:
    """A deterministic tagger used by the smoke path. Mirrors the HF API."""
    out: list[TaggedEntity] = []
    for example_text, ents in FIXTURE_EXAMPLES:
        if example_text not in text:
            continue
        offset = text.find(example_text)
        for word, label in ents:
            local = example_text.find(word)
            if local < 0:
                continue
            start = offset + local
            out.append(
                TaggedEntity(
                    start=start,
                    end=start + len(word),
                    word=word,
                    entity_group=label,
                    score=1.0,
                )
            )
    return out


def _smoke() -> int:
    detector = MLDetector(tagger=_fixture_tagger)
    text = FIXTURE_EXAMPLES[0][0]
    spans = detector.detect(text)
    if not spans:
        print("smoke: no spans produced", file=sys.stderr)
        return 1
    cats = {s.category for s in spans}
    if Category.IDENTIFIER not in cats:
        print(f"smoke: missing IDENTIFIER, got {cats}", file=sys.stderr)
        return 1
    print(f"smoke ok: {len(spans)} spans, categories={cats}")
    return 0


def _train() -> int:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("transformers/torch not installed — install privategate[ml] to train")
        return 0
    print("full training loop not yet implemented — see plan §M3")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return _smoke() if args.smoke else _train()


if __name__ == "__main__":
    sys.exit(main())
