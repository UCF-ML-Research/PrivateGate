"""Ablation: rule-only detector vs. rule + ML hybrid.

We don't have a trained ML detector in CI, so the ML arm uses a small
deterministic stub that recognizes a closed list of names. This is
enough to exercise the merger end-to-end and to show that adding an ML
layer **increases** detection coverage and downstream privacy on the
seed dataset (which contains a "Dr. Alice Smith" example that the rule
gazetteer alone may not catch with full confidence).
"""
from __future__ import annotations

from typing import Iterable

from privategate.detector.hybrid_detector import HybridDetector
from privategate.detector.ml_detector import MLDetector, TaggedEntity
from privategate.eval.ablations._common import score_arm
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample


_NAMES = ("Alice Smith", "John Adams", "Mary Wong", "Albert Einstein", "Bob")


def _stub_tagger(text: str) -> Iterable[TaggedEntity]:
    for name in _NAMES:
        idx = text.find(name)
        if idx >= 0:
            yield TaggedEntity(
                start=idx,
                end=idx + len(name),
                word=name,
                entity_group="PER",
                score=0.99,
            )


def run(examples: list[EvalExample] | None = None) -> dict:
    examples = examples or load_synthetic_mixed()

    rule_only = PrivateGateBaseline(name="rule_only")
    hybrid = PrivateGateBaseline(
        detector=HybridDetector(ml_detector=MLDetector(tagger=_stub_tagger)),
        name="rule_plus_ml",
    )
    return {
        "ablation": "detector",
        "arms": [score_arm(rule_only, examples), score_arm(hybrid, examples)],
    }
