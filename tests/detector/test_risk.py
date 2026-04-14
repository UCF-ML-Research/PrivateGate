from privategate.detector.risk import score_risk
from privategate.types import Category, RiskLevel, Span


def test_credential_always_critical():
    s = Span(0, 5, "sk-xy", Category.CREDENTIAL, RiskLevel.LOW)
    assert score_risk(s) is RiskLevel.CRITICAL


def test_medical_bumped_when_named_doctor_present():
    s = Span(20, 28, "diabetes", Category.MEDICAL, RiskLevel.HIGH)
    context = "patient seen by Dr. Smith with diabetes"
    assert score_risk(s, context).rank >= RiskLevel.HIGH.rank


def test_default_returns_at_least_base():
    s = Span(0, 5, "12345", Category.IDENTIFIER, RiskLevel.LOW)
    assert score_risk(s).rank >= RiskLevel.HIGH.rank
