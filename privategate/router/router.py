"""Router. Combines policy-driven routing with the semantic-dependency probe.

A query enters the **secure** path if either:
  1. the rewriter inserted a secure-slot placeholder (i.e. the policy
     engine already escalated a span), or
  2. a semantic-dependency probe flagged a span as semantic-critical.

The router does not run the probe itself — that's an injected callable
so the heavy proxy model stays out of the unit test path. `decide_probe_targets`
selects which spans deserve a probe (high-risk, not already secure-slot).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

from privategate.types import (
    Action,
    PolicyDecision,
    ProbeResult,
    RewriteResult,
    RiskLevel,
    RoutingDecision,
    Span,
)

ProbeRunner = Callable[[Span], ProbeResult]


class Router:
    def __init__(self, probe_min_risk: RiskLevel = RiskLevel.HIGH) -> None:
        self._probe_min_risk = probe_min_risk

    def route(
        self,
        rewrite_result: RewriteResult,
        probe_results: Sequence[ProbeResult] = (),
    ) -> RoutingDecision:
        if rewrite_result.has_secure_slots:
            return RoutingDecision(path="secure", reason="contains secure-slot placeholders")
        critical = [p for p in probe_results if p.semantic_critical]
        if critical:
            cats = ",".join(sorted({p.span.category.value for p in critical}))
            return RoutingDecision(
                path="secure",
                reason=f"semantic-dependency probe escalated ({cats})",
            )
        return RoutingDecision(path="standard", reason="no critical spans")

    def decide_probe_targets(
        self,
        decisions: Iterable[PolicyDecision],
    ) -> list[Span]:
        """Return the spans the caller should run a probe on.

        We probe spans that are risky enough to matter but were *not*
        already escalated to ``secure-slot`` — those are the ambiguous
        cases where masking might silently break the task.
        """
        targets: list[Span] = []
        for d in decisions:
            if d.action is Action.SECURE_SLOT:
                continue
            if d.span.risk.rank >= self._probe_min_risk.rank:
                targets.append(d.span)
        return targets

    def run_probes(
        self,
        decisions: Iterable[PolicyDecision],
        runner: ProbeRunner,
    ) -> list[ProbeResult]:
        return [runner(span) for span in self.decide_probe_targets(decisions)]
