import pytest

from privategate.rewriter.actions import apply_abstract
from privategate.types import Category, RiskLevel, Span


def _med_span(text: str) -> Span:
    return Span(0, len(text), text, Category.MEDICAL, RiskLevel.HIGH)


def test_low_returns_original_text():
    assert apply_abstract(_med_span("type 2 diabetes"), level="low") == "type 2 diabetes"


def test_medium_returns_superclass():
    out = apply_abstract(_med_span("type 2 diabetes"), level="medium")
    assert "diabetes" not in out.lower()
    assert "chronic" in out.lower()


def test_high_returns_category_mask():
    assert apply_abstract(_med_span("anything"), level="high") == "[MEDICAL]"


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        apply_abstract(_med_span("x"), level="weird")
