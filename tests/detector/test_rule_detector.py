from privategate.detector.rule_detector import RuleDetector
from privategate.types import Category


def test_detects_multiple_categories():
    text = "I'm Dr. Bob Jones, my SSN is 123-45-6789 and I have type 2 diabetes"
    spans = RuleDetector().detect(text)
    cats = {s.category for s in spans}
    assert Category.IDENTIFIER in cats
    assert Category.MEDICAL in cats


def test_no_overlap_in_output():
    text = "card 4111-1111-1111-1111 ssn 123-45-6789"
    spans = RuleDetector().detect(text)
    sorted_spans = sorted(spans, key=lambda s: s.start)
    for a, b in zip(sorted_spans, sorted_spans[1:]):
        assert a.end <= b.start, f"overlapping spans: {a} vs {b}"


def test_empty_input():
    assert RuleDetector().detect("") == []
