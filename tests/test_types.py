import pytest

from privategate.types import (
    Action,
    Category,
    PolicyDecision,
    RewriteResult,
    RiskLevel,
    Span,
)


def test_risk_rank_ordering():
    assert RiskLevel.LOW.rank < RiskLevel.MEDIUM.rank < RiskLevel.HIGH.rank < RiskLevel.CRITICAL.rank


def test_span_validates_offsets():
    with pytest.raises(ValueError):
        Span(start=5, end=5, text="x", category=Category.IDENTIFIER, risk=RiskLevel.LOW)
    with pytest.raises(ValueError):
        Span(start=0, end=1, text="", category=Category.IDENTIFIER, risk=RiskLevel.LOW)


def test_span_overlap():
    a = Span(0, 5, "hello", Category.IDENTIFIER, RiskLevel.LOW)
    b = Span(3, 8, "lo wo", Category.IDENTIFIER, RiskLevel.LOW)
    c = Span(6, 11, "world", Category.IDENTIFIER, RiskLevel.LOW)
    assert a.overlaps(b)
    assert not a.overlaps(c)


def test_rewrite_result_defaults():
    r = RewriteResult(
        original_text="hi",
        transformed_text="hi",
        placeholder_map={},
        decisions=[],
    )
    assert r.has_secure_slots is False


def test_action_enum_values():
    assert Action.SECURE_SLOT.value == "secure-slot"
    assert Action.MASK.value == "mask"
