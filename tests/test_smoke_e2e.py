from privategate.detector.rule_detector import RuleDetector
from privategate.policy.engine import PolicyEngine
from privategate.policy.table import load_default_policy
from privategate.rewriter.rewriter import Rewriter


def test_end_to_end_no_pii():
    detector = RuleDetector()
    engine = PolicyEngine(load_default_policy())
    rewriter = Rewriter()

    text = "what is the capital of france"
    spans = detector.detect(text)
    decisions = engine.decide(spans)
    result = rewriter.rewrite(text, decisions)

    assert result.transformed_text == text
    assert result.placeholder_map == {}


def test_end_to_end_with_ssn():
    detector = RuleDetector()
    engine = PolicyEngine(load_default_policy())
    rewriter = Rewriter()

    text = "my SSN is 123-45-6789, please help"
    spans = detector.detect(text)
    decisions = engine.decide(spans)
    result = rewriter.rewrite(text, decisions)

    assert "123-45-6789" not in result.transformed_text
    assert any("123-45-6789" in v for v in result.placeholder_map.values())
