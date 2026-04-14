"""Ablation: with vs. without the semantic-dependency probe.

Both arms use the default policy and rule detector. The "with-probe"
arm injects an echo proxy model whose answers diverge whenever a span
is removed (because the divergence metric defaults to token-Jaccard, an
echo proxy makes divergence equal to the difference between the
original and the masked/abstracted variants — i.e. it directly reflects
how much the span carried lexical content).
"""
from __future__ import annotations

from privategate.eval.ablations._common import score_arm
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample


def _echo_proxy(text: str) -> str:
    return text


def run(examples: list[EvalExample] | None = None) -> dict:
    examples = examples or load_synthetic_mixed()
    no_probe = PrivateGateBaseline(name="no_probe")
    with_probe = PrivateGateBaseline(
        probe_proxy=_echo_proxy,
        probe_threshold=0.3,
        name="with_probe",
    )
    return {
        "ablation": "semantic_probe",
        "arms": [score_arm(no_probe, examples), score_arm(with_probe, examples)],
    }
