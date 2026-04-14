from privategate.detector.hybrid_detector import HybridDetector
from privategate.detector.ml_detector import MLDetector, TaggedEntity
from privategate.types import Category


def test_hybrid_without_ml_equals_rule_only():
    h = HybridDetector()
    spans = h.detect("ssn 123-45-6789")
    assert len(spans) == 1
    assert h.has_ml is False


def test_hybrid_combines_rule_and_ml_spans():
    def tagger(text: str):
        idx = text.find("Alice")
        if idx >= 0:
            yield TaggedEntity(start=idx, end=idx + 5, word="Alice", entity_group="PER", score=0.99)

    h = HybridDetector(ml_detector=MLDetector(tagger=tagger))
    spans = h.detect("Alice's ssn is 123-45-6789")
    cats = {s.category for s in spans}
    assert Category.IDENTIFIER in cats
    assert any("123-45-6789" in s.text for s in spans)
    assert any("Alice" in s.text for s in spans)


def test_hybrid_resolves_overlap_with_rule_winner():
    def tagger(text: str):
        yield TaggedEntity(start=0, end=11, word="123-45-6789", entity_group="ID", score=0.9)

    h = HybridDetector(ml_detector=MLDetector(tagger=tagger))
    spans = h.detect("123-45-6789")
    assert len(spans) == 1
    assert spans[0].source == "rule"
