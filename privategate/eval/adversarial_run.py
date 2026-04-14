"""Adversarial results-table runner.

For every baseline × every attacker, run the attacker on each
example's outbound payload and report the recovery rate against the
example's gold spans. The output is a JSON-serialisable dict the report
(M7) can paste into the adversarial table.
"""
from __future__ import annotations

import json
from typing import Sequence

from privategate.adversary.base import Attacker
from privategate.adversary.embedding_inversion import EmbeddingInverter
from privategate.adversary.llm_guesser import LLMGuesser
from privategate.adversary.scorer import RecoveryReport, recovery_rate
from privategate.eval.baselines.base import Baseline, BaselineRun
from privategate.eval.baselines.full_abstract import FullAbstractBaseline
from privategate.eval.baselines.full_mask import FullMaskBaseline
from privategate.eval.baselines.full_secure import FullSecureBaseline
from privategate.eval.baselines.plaintext import PlaintextBaseline
from privategate.eval.baselines.privategate import PrivateGateBaseline
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.eval.example import EvalExample


def _aggregate(reports: list[RecoveryReport]) -> RecoveryReport:
    total_gold = sum(r.n_gold for r in reports)
    total_recovered = sum(r.n_recovered for r in reports)
    per_cat_gold: dict[str, int] = {}
    per_cat_rec: dict[str, int] = {}
    for r in reports:
        for cat, stats in r.by_category.items():
            per_cat_gold[cat] = per_cat_gold.get(cat, 0) + stats["n_gold"]
            per_cat_rec[cat] = per_cat_rec.get(cat, 0) + stats["n_recovered"]
    by_cat = {
        cat: {
            "rate": (per_cat_rec[cat] / g) if g else 0.0,
            "n_gold": g,
            "n_recovered": per_cat_rec[cat],
        }
        for cat, g in per_cat_gold.items()
    }
    return RecoveryReport(
        overall_rate=(total_recovered / total_gold) if total_gold else 0.0,
        n_gold=total_gold,
        n_recovered=total_recovered,
        by_category=by_cat,
    )


def _attack_baseline(
    examples: Sequence[EvalExample],
    runs: Sequence[BaselineRun],
    attacker: Attacker,
) -> RecoveryReport:
    per_example: list[RecoveryReport] = []
    for ex, run in zip(examples, runs):
        result = attacker.attack(run.outbound_text)
        per_example.append(recovery_rate(result.predictions, ex.spans))
    return _aggregate(per_example)


def default_attackers() -> list[Attacker]:
    return [EmbeddingInverter(), LLMGuesser()]


def default_baselines() -> list[Baseline]:
    return [
        PlaintextBaseline(),
        FullMaskBaseline(),
        FullAbstractBaseline(),
        FullSecureBaseline(),
        PrivateGateBaseline(),
    ]


def run_adversarial_table(
    examples: Sequence[EvalExample] | None = None,
    baselines: Sequence[Baseline] | None = None,
    attackers: Sequence[Attacker] | None = None,
) -> dict:
    examples = list(examples) if examples is not None else load_synthetic_mixed()
    baselines = list(baselines) if baselines is not None else default_baselines()
    attackers = list(attackers) if attackers is not None else default_attackers()

    table: list[dict] = []
    for baseline in baselines:
        runs = baseline.run_all(examples)
        for attacker in attackers:
            report = _attack_baseline(examples, runs, attacker)
            row = {
                "baseline": baseline.name,
                "attacker": attacker.name,
                **report.to_dict(),
            }
            table.append(row)

    return {"n_examples": len(examples), "results": table}


def main() -> int:
    print(json.dumps(run_adversarial_table(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
