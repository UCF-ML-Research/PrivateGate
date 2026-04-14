from privategate.detector.rules import detect_credit_card
from privategate.types import Category, RiskLevel


def test_detects_valid_visa():
    spans = detect_credit_card("My card is 4111 1111 1111 1111 ok?")
    assert len(spans) == 1
    assert spans[0].category is Category.FINANCIAL
    assert spans[0].risk is RiskLevel.CRITICAL


def test_rejects_invalid_luhn():
    assert detect_credit_card("4111 1111 1111 1112") == []


def test_no_false_positive_short_number():
    assert detect_credit_card("call 1234") == []
