"""Shared scoring helpers for ablation runs.

Every ablation arm produces the same shape of metrics so they can be
compared in a single table. The functions here take an instantiated
baseline + dataset and produce a dict of {privacy, utility, efficiency,
adversarial} metrics — exactly what each ablation row needs to report.
"""
from __future__ import annotations

from typing import Sequence

from privategate.adversary.base import Attacker
from privategate.adversary.embedding_inversion import EmbeddingInverter
from privategate.adversary.llm_guesser import LLMGuesser
from privategate.adversary.scorer import recovery_rate
from privategate.eval.baselines.base import Baseline, BaselineRun
from privategate.eval.example import EvalExample
from privategate.eval.metrics.efficiency import (
    latency_percentiles,
    secure_path_fraction,
)
from privategate.eval.metrics.privacy import (
    category_weighted_leakage,
    span_exposure_rate,
)
from privategate.eval.metrics.utility import answer_similarity, exact_match


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_arm(
    baseline: Baseline,
    examples: Sequence[EvalExample],
    attackers: Sequence[Attacker] | None = None,
) -> dict:
    runs: list[BaselineRun] = baseline.run_all(examples)
    attackers = list(attackers) if attackers is not None else [
        EmbeddingInverter(),
        LLMGuesser(),
    ]

    privacy: list[float] = []
    weighted: list[float] = []
    em: list[float] = []
    sim: list[float] = []
    for ex, run in zip(examples, runs):
        privacy.append(span_exposure_rate(run.outbound_text, ex.spans))
        weighted.append(category_weighted_leakage(run.outbound_text, ex.spans))
        if ex.reference_answer is not None:
            em.append(exact_match(run.answer, ex.reference_answer))
            sim.append(answer_similarity(run.answer, ex.reference_answer))

    latency = latency_percentiles([r.latency_ms for r in runs])
    secure_frac = secure_path_fraction([r.routing_path for r in runs])

    adversarial: dict[str, float] = {}
    for attacker in attackers:
        recovered = 0
        total = 0
        for ex, run in zip(examples, runs):
            res = attacker.attack(run.outbound_text)
            r = recovery_rate(res.predictions, ex.spans)
            recovered += r.n_recovered
            total += r.n_gold
        adversarial[attacker.name] = (recovered / total) if total else 0.0

    return {
        "name": baseline.name,
        "privacy": {
            "span_exposure_rate": round(_avg(privacy), 4),
            "category_weighted_leakage": round(_avg(weighted), 4),
        },
        "utility": {
            "exact_match": round(_avg(em), 4),
            "answer_similarity": round(_avg(sim), 4),
        },
        "efficiency": {
            "latency_p50_ms": round(latency.p50, 4),
            "latency_p95_ms": round(latency.p95, 4),
            "secure_path_fraction": round(secure_frac, 4),
        },
        "adversarial": {k: round(v, 4) for k, v in adversarial.items()},
        "n": latency.n,
    }
