from privategate.detector.merge import merge_spans
from privategate.types import Category, RiskLevel, Span


def _span(start, end, text, cat, risk, source="rule"):
    return Span(start, end, text, cat, risk, source=source)


def test_rule_wins_over_ml_on_overlap():
    rule = [_span(0, 11, "123-45-6789", Category.IDENTIFIER, RiskLevel.HIGH, "rule")]
    ml = [_span(4, 11, "5-6789", Category.IDENTIFIER, RiskLevel.HIGH, "ml")]
    out = merge_spans(rule, ml)
    assert len(out) == 1
    assert out[0].source == "rule"


def test_disjoint_spans_kept():
    rule = [_span(0, 5, "alice", Category.IDENTIFIER, RiskLevel.HIGH, "rule")]
    ml = [_span(10, 15, "paris", Category.PERSONAL_CONTEXT, RiskLevel.MEDIUM, "ml")]
    out = merge_spans(rule, ml)
    assert len(out) == 2


def test_higher_risk_wins_within_same_source():
    a = [_span(0, 5, "abcde", Category.IDENTIFIER, RiskLevel.MEDIUM, "ml")]
    b = [_span(0, 5, "abcde", Category.IDENTIFIER, RiskLevel.HIGH, "ml")]
    out = merge_spans(a, b)
    assert len(out) == 1
    assert out[0].risk is RiskLevel.HIGH


def test_output_is_sorted_and_nonoverlapping():
    rule = [
        _span(20, 25, "world", Category.IDENTIFIER, RiskLevel.HIGH, "rule"),
        _span(0, 5, "hello", Category.IDENTIFIER, RiskLevel.HIGH, "rule"),
    ]
    out = merge_spans(rule, [])
    assert [s.start for s in out] == [0, 20]


def test_empty_inputs():
    assert merge_spans([], []) == []
