"""Ablation: strict vs. balanced (default) vs. permissive policy table."""
from __future__ import annotations

from privategate.eval.ablations._common import score_arm
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample
from privategate.policy.table import (
    load_default_policy,
    load_permissive_policy,
    load_strict_policy,
)


def run(examples: list[EvalExample] | None = None) -> dict:
    examples = examples or load_synthetic_mixed()
    arms = [
        PrivateGateBaseline(policy_table=load_strict_policy(), name="strict"),
        PrivateGateBaseline(policy_table=load_default_policy(), name="balanced"),
        PrivateGateBaseline(policy_table=load_permissive_policy(), name="permissive"),
    ]
    return {
        "ablation": "policy_table",
        "arms": [score_arm(b, examples) for b in arms],
    }
