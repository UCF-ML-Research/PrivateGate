from privategate.rewriter.actions import apply_abstract
from privategate.types import Category, RiskLevel, Span


def test_known_medical_term_abstracted():
    s = Span(0, 28, "metastatic pancreatic cancer", Category.MEDICAL, RiskLevel.HIGH)
    out = apply_abstract(s)
    assert "cancer" not in out.lower()
    assert "serious" in out.lower()


def test_unknown_medical_term_uses_fallback():
    s = Span(0, 12, "rare disease", Category.MEDICAL, RiskLevel.HIGH)
    out = apply_abstract(s)
    assert "health" in out.lower()


def test_personal_context_uses_fallback():
    s = Span(0, 5, "lives", Category.PERSONAL_CONTEXT, RiskLevel.HIGH)
    assert apply_abstract(s) == "personal context"
