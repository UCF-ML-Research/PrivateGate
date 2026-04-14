from privategate.detector.rules import detect_api_key
from privategate.types import Category, RiskLevel


def test_detects_sk_key():
    spans = detect_api_key("export OPENAI_KEY=sk-abcd1234efgh5678ijkl")
    assert len(spans) == 1
    assert spans[0].category is Category.CREDENTIAL
    assert spans[0].risk is RiskLevel.CRITICAL


def test_detects_token_form():
    spans = detect_api_key("token_ABCDEFGHIJKLMNOPQRST")
    assert len(spans) == 1


def test_no_match_short_string():
    assert detect_api_key("sk-tiny") == []
