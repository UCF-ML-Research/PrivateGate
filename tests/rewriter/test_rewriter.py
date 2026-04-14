from privategate.detector.rule_detector import RuleDetector
from privategate.policy.engine import PolicyEngine
from privategate.policy.table import load_default_policy
from privategate.rewriter.rewriter import Rewriter


def _pipeline(text: str):
    spans = RuleDetector().detect(text)
    decisions = PolicyEngine(load_default_policy()).decide(spans)
    return Rewriter(faker_seed=0).rewrite(text, decisions)


def test_ssn_does_not_appear_in_transformed_text():
    result = _pipeline("My SSN is 123-45-6789 please help")
    assert "123-45-6789" not in result.transformed_text
    assert "123-45-6789" in result.placeholder_map.values() \
        or any("123-45-6789" in v for v in result.placeholder_map.values())


def test_api_key_routed_to_secure_slot():
    result = _pipeline("token=sk-abcdefghij1234567890 please rotate")
    assert "sk-abcdefghij1234567890" not in result.transformed_text
    assert result.has_secure_slots is True
    assert any(v == "sk-abcdefghij1234567890" for v in result.placeholder_map.values())


def test_no_pii_text_unchanged():
    result = _pipeline("what is the capital of france")
    assert result.transformed_text == "what is the capital of france"
    assert result.has_secure_slots is False
    assert result.placeholder_map == {}


def test_medical_term_abstracted():
    result = _pipeline("I have type 2 diabetes and need diet advice")
    assert "diabetes" not in result.transformed_text.lower()
    assert "chronic condition" in result.transformed_text.lower()
