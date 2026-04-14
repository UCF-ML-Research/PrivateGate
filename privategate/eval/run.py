"""Main results-table runner.

Runs every baseline against a dataset, scores each run with the privacy /
utility / efficiency metrics, and returns a nested dict the caller can
print or serialize. Plan §6 calls for this to be reproducible from a
single command — the `main()` entry point at the bottom does exactly
that on the SyntheticMixed seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from privategate.eval.baselines.base import Baseline, BaselineRun
from privategate.eval.baselines.full_abstract import FullAbstractBaseline
from privategate.eval.baselines.full_mask import FullMaskBaseline
from privategate.eval.baselines.full_secure import FullSecureBaseline
from privategate.eval.baselines.plaintext import PlaintextBaseline
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample
from privategate.eval.metrics.efficiency import (
    LatencyReport,
    latency_percentiles,
    secure_path_fraction,
)
from privategate.eval.metrics.privacy import (
    category_weighted_leakage,
    span_exposure_rate,
)
from privategate.eval.metrics.utility import answer_similarity, exact_match


@dataclass
class BaselineResults:
    name: str
    span_exposure: float
    weighted_leakage: float
    exact_match: float
    answer_similarity: float
    latency: LatencyReport
    secure_fraction: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "privacy": {
                "span_exposure_rate": round(self.span_exposure, 4),
                "category_weighted_leakage": round(self.weighted_leakage, 4),
            },
            "utility": {
                "exact_match": round(self.exact_match, 4),
                "answer_similarity": round(self.answer_similarity, 4),
            },
            "efficiency": {
                "latency_p50_ms": round(self.latency.p50, 4),
                "latency_p95_ms": round(self.latency.p95, 4),
                "latency_mean_ms": round(self.latency.mean, 4),
                "secure_path_fraction": round(self.secure_fraction, 4),
            },
            "n": self.latency.n,
        }


def _score(baseline: Baseline, examples: Sequence[EvalExample]) -> BaselineResults:
    runs: list[BaselineRun] = baseline.run_all(examples)

    privacy_scores: list[float] = []
    weighted_scores: list[float] = []
    em_scores: list[float] = []
    sim_scores: list[float] = []

    for ex, run in zip(examples, runs):
        privacy_scores.append(span_exposure_rate(run.outbound_text, ex.spans))
        weighted_scores.append(category_weighted_leakage(run.outbound_text, ex.spans))
        if ex.reference_answer is not None:
            em_scores.append(exact_match(run.answer, ex.reference_answer))
            sim_scores.append(answer_similarity(run.answer, ex.reference_answer))

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return BaselineResults(
        name=baseline.name,
        span_exposure=_avg(privacy_scores),
        weighted_leakage=_avg(weighted_scores),
        exact_match=_avg(em_scores),
        answer_similarity=_avg(sim_scores),
        latency=latency_percentiles([r.latency_ms for r in runs]),
        secure_fraction=secure_path_fraction([r.routing_path for r in runs]),
    )


def default_baselines() -> list[Baseline]:
    return [
        PlaintextBaseline(),
        FullMaskBaseline(),
        FullAbstractBaseline(),
        FullSecureBaseline(),
        PrivateGateBaseline(),
    ]


def run_main_table(
    examples: Sequence[EvalExample] | None = None,
    baselines: Sequence[Baseline] | None = None,
) -> dict:
    examples = list(examples) if examples is not None else load_synthetic_mixed()
    baselines = list(baselines) if baselines is not None else default_baselines()
    table = [_score(b, examples).to_dict() for b in baselines]
    return {
        "n_examples": len(examples),
        "results": table,
    }


def main() -> int:
    print(json.dumps(run_main_table(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
