from privategate.policy.engine import PolicyEngine
from privategate.policy.table import load_default_policy
from privategate.router.router import Router
from privategate.types import (
    Action,
    Category,
    PolicyDecision,
    ProbeResult,
    RewriteResult,
    RiskLevel,
    Span,
)


def _rw(has_secure: bool) -> RewriteResult:
    return RewriteResult(
        original_text="x",
        transformed_text="x",
        placeholder_map={},
        decisions=[],
        has_secure_slots=has_secure,
    )


def _span(cat: Category, risk: RiskLevel) -> Span:
    return Span(0, 5, "xxxxx", cat, risk)


def test_routes_to_secure_on_secure_slot():
    decision = Router().route(_rw(True))
    assert decision.path == "secure"


def test_routes_to_standard_when_clean():
    decision = Router().route(_rw(False))
    assert decision.path == "standard"


def test_routes_to_secure_when_probe_flags_critical():
    span = _span(Category.MEDICAL, RiskLevel.HIGH)
    probes = [ProbeResult(span=span, divergence=0.7, semantic_critical=True)]
    decision = Router().route(_rw(False), probe_results=probes)
    assert decision.path == "secure"
    assert "MEDICAL" in decision.reason


def test_decide_probe_targets_skips_secure_slot_spans():
    eng = PolicyEngine(load_default_policy())
    spans = [
        _span(Category.MEDICAL, RiskLevel.HIGH),       # high -> abstract -> probed
        _span(Category.CREDENTIAL, RiskLevel.CRITICAL),  # critical -> secure-slot -> not probed
        _span(Category.PERSONAL_CONTEXT, RiskLevel.LOW),  # low -> not probed
    ]
    decisions = eng.decide(spans)
    targets = Router().decide_probe_targets(decisions)
    assert len(targets) == 1
    assert targets[0].category is Category.MEDICAL


def test_run_probes_invokes_runner_for_each_target():
    eng = PolicyEngine(load_default_policy())
    decisions = eng.decide([_span(Category.MEDICAL, RiskLevel.HIGH)])
    seen: list[Span] = []

    def runner(span):
        seen.append(span)
        return ProbeResult(span=span, divergence=0.1, semantic_critical=False)

    results = Router().run_probes(decisions, runner)
    assert len(results) == 1
    assert len(seen) == 1
