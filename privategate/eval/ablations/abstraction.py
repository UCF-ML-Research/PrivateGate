"""Ablation: abstraction-level knob (low / medium / high)."""
from __future__ import annotations

from privategate.eval.ablations._common import score_arm
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample


def run(examples: list[EvalExample] | None = None) -> dict:
    examples = examples or load_synthetic_mixed()
    arms = [
        PrivateGateBaseline(abstraction_level=level, name=f"abstract_{level}")
        for level in ("low", "medium", "high")
    ]
    return {
        "ablation": "abstraction_level",
        "arms": [score_arm(b, examples) for b in arms],
    }
