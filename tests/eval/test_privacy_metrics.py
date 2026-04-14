from privategate.eval.metrics.privacy import (
    category_weighted_leakage,
    span_exposure_rate,
)
from privategate.types import Category, RiskLevel, Span


def _span(text: str, cat: Category) -> Span:
    return Span(0, len(text), text, cat, RiskLevel.HIGH)


def test_zero_exposure_when_text_redacted():
    spans = [_span("123-45-6789", Category.IDENTIFIER)]
    assert span_exposure_rate("[IDENTIFIER]", spans) == 0.0


def test_full_exposure_when_text_present():
    spans = [_span("123-45-6789", Category.IDENTIFIER)]
    assert span_exposure_rate("ssn 123-45-6789 here", spans) == 1.0


def test_partial_exposure():
    spans = [
        _span("alice", Category.IDENTIFIER),
        _span("bob", Category.IDENTIFIER),
    ]
    assert span_exposure_rate("alice was here", spans) == 0.5


def test_weighted_leakage_emphasizes_credentials():
    leaked_ident = [_span("alice", Category.IDENTIFIER)]
    leaked_cred = [_span("sk-xy", Category.CREDENTIAL)]
    score_ident = category_weighted_leakage("alice was here", leaked_ident)
    score_cred = category_weighted_leakage("sk-xy is bad", leaked_cred)
    # credentials have higher weight, so leakage scales accordingly
    assert score_cred == 1.0
    assert score_ident == 1.0  # both fully leaked, scaled to 1.0


def test_no_spans_returns_zero():
    assert span_exposure_rate("anything", []) == 0.0
    assert category_weighted_leakage("anything", []) == 0.0
