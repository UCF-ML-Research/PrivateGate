from privategate.router.probe import three_variant_probe
from privategate.types import Category, RiskLevel, Span


def _medical_span(query: str, term: str) -> Span:
    start = query.find(term)
    return Span(start, start + len(term), term, Category.MEDICAL, RiskLevel.HIGH)


def test_probe_flags_critical_when_answer_changes():
    """Proxy returns very different answers depending on whether the medical
    term is present — the span is semantic-critical."""
    query = "I have type 2 diabetes, what diet should I follow?"

    def proxy(text: str) -> str:
        if "type 2 diabetes" in text.lower():
            return "low carb mediterranean diet metformin glucose monitoring"
        return "general healthy eating advice"

    span = _medical_span(query, "type 2 diabetes")
    result = three_variant_probe(query, span, proxy)
    assert result.semantic_critical is True
    assert result.divergence > 0.3


def test_probe_does_not_flag_when_answer_unchanged():
    query = "I have type 2 diabetes, what is the capital of France?"
    constant = "the capital of france is paris"

    def proxy(text: str) -> str:
        return constant

    span = _medical_span(query, "type 2 diabetes")
    result = three_variant_probe(query, span, proxy)
    assert result.semantic_critical is False
    assert result.divergence == 0.0


def test_probe_uses_custom_embedder():
    query = "I have diabetes today"

    def proxy(text: str) -> str:
        return "answer for: " + text

    def embedder(text: str):
        # all answers map to the same point — divergence is always 0
        return [1.0, 0.0, 0.0]

    span = _medical_span(query, "diabetes")
    result = three_variant_probe(query, span, proxy, embedder=embedder)
    assert result.divergence == 0.0
    assert result.semantic_critical is False


def test_probe_rejects_out_of_range_span():
    import pytest

    span = Span(100, 105, "hello", Category.MEDICAL, RiskLevel.HIGH)
    with pytest.raises(ValueError):
        three_variant_probe("short text", span, lambda t: t)


def test_probe_threshold_is_respected():
    query = "patient has diabetes"

    def proxy(text: str) -> str:
        return "diabetes" if "diabetes" in text else "no condition"

    span = _medical_span(query, "diabetes")
    # divergence is exactly 1.0 here (fully disjoint token sets); a threshold
    # >= 1.0 must NOT flag because the probe uses a strict `>` comparison.
    high_threshold = three_variant_probe(query, span, proxy, threshold=1.0)
    low_threshold = three_variant_probe(query, span, proxy, threshold=0.05)
    assert low_threshold.semantic_critical is True
    assert high_threshold.semantic_critical is False
