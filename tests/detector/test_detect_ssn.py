from privategate.detector.rules import detect_ssn
from privategate.types import Category, RiskLevel


def test_detects_dashed_ssn():
    spans = detect_ssn("My SSN is 123-45-6789.")
    assert len(spans) == 1
    s = spans[0]
    assert s.text == "123-45-6789"
    assert s.category is Category.IDENTIFIER
    assert s.risk is RiskLevel.HIGH


def test_detects_spaced_ssn():
    spans = detect_ssn("ssn 123 45 6789")
    assert len(spans) == 1
    assert spans[0].text == "123 45 6789"


def test_rejects_invalid_area():
    assert detect_ssn("000-12-3456") == []
    assert detect_ssn("666-12-3456") == []
    assert detect_ssn("900-12-3456") == []


def test_no_match_in_plain_text():
    assert detect_ssn("the answer is forty-two") == []
