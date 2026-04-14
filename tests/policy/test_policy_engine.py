from privategate.policy.engine import PolicyEngine
from privategate.policy.table import load_default_policy
from privategate.types import Action, Category, RiskLevel, Span


def _span(cat: Category, risk: RiskLevel) -> Span:
    return Span(0, 5, "xxxxx", cat, risk)


def test_credential_critical_secure_slot():
    eng = PolicyEngine(load_default_policy())
    decisions = eng.decide([_span(Category.CREDENTIAL, RiskLevel.CRITICAL)])
    assert decisions[0].action is Action.SECURE_SLOT


def test_identifier_high_pseudonymize():
    eng = PolicyEngine(load_default_policy())
    decisions = eng.decide([_span(Category.IDENTIFIER, RiskLevel.HIGH)])
    assert decisions[0].action is Action.PSEUDONYMIZE


def test_needs_secure_path_flag():
    eng = PolicyEngine(load_default_policy())
    d_safe = eng.decide([_span(Category.PERSONAL_CONTEXT, RiskLevel.LOW)])
    d_critical = eng.decide([_span(Category.CREDENTIAL, RiskLevel.CRITICAL)])
    assert eng.needs_secure_path(d_safe) is False
    assert eng.needs_secure_path(d_critical) is True
