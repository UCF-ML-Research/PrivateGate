from __future__ import annotations

from privategate.policy.engine import PolicyEngine
from privategate.policy.reference_rules import REFERENCE_RULES, REFERENCE_STRICTNESS
from privategate.types import Context, Mode, Request, Signal

from ..baselines.reference_r1r4 import OracleR1R4
from ..harness.runner import run_eval
from ..harness.schemas import GoldItem


_ALL_NEG = {
    "none": True, "pii": False, "phi": False, "pci": False, "secret": False,
    "ip_confidential": False, "regulated_eu": False, "regulated_us": False,
    "injection": False,
}


def _sig(cat: str, severity: int = 0, confidence: float = 0.9) -> Signal:
    return Signal(
        type=f"category.{cat}",
        score=confidence,
        source="test",
        evidence={"severity": severity},
    )


def _engine() -> PolicyEngine:
    return PolicyEngine(REFERENCE_RULES, REFERENCE_STRICTNESS)


def _req(text: str = "x") -> Request:
    return Request(payload=text, context=Context())


def test_no_signals_defaults_to_plaintext():
    d = _engine().evaluate(_req(), [])
    assert d.mode == Mode.PLAINTEXT


def test_injection_wins_over_phi_severity_three():
    d = _engine().evaluate(_req(), [_sig("phi", 3), _sig("injection", 0)])
    assert d.mode == Mode.ABSTAIN
    assert d.matched_rule == "R1_injection"


def test_severity_three_without_injection_routes_to_he():
    d = _engine().evaluate(_req(), [_sig("phi", 3)])
    assert d.mode == Mode.HE_SMPC
    assert d.matched_rule == "R2_severity_three"


def test_any_sensitive_routes_to_perturb():
    d = _engine().evaluate(_req(), [_sig("pii", 2)])
    assert d.mode == Mode.PERTURB
    assert d.matched_rule == "R3_any_sensitive"


def _gi(id_: str, gold: Mode, cats_true: dict[str, bool] | None = None,
        sev: dict[str, int] | None = None) -> GoldItem:
    cats = dict(_ALL_NEG)
    if cats_true:
        for k, v in cats_true.items():
            cats[k] = v
        cats["none"] = not any(v for k, v in cats.items() if k != "none")
    return GoldItem(
        id=id_,
        prompt="x",
        source="synthetic",
        categories=cats,
        severity=sev or {},
        spans=[],
        gold_mode=gold,
    )


def test_oracle_matches_gold_on_canonical_items():
    items = [
        _gi("a", Mode.PLAINTEXT),
        _gi("b", Mode.PERTURB, {"pii": True}, {"pii": 2}),
        _gi("c", Mode.HE_SMPC, {"phi": True}, {"phi": 3}),
        _gi("d", Mode.ABSTAIN, {"injection": True}),
    ]
    out = run_eval(OracleR1R4(), items)
    assert out.fdr_overall == 0.0
    assert out.oer == 0.0
