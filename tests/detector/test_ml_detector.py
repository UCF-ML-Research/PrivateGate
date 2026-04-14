from privategate.detector.ml_detector import MLDetector, TaggedEntity
from privategate.types import Category, RiskLevel


def _fake_tagger(text: str):
    yield TaggedEntity(start=0, end=5, word="Alice", entity_group="PER", score=0.99)
    yield TaggedEntity(start=15, end=20, word="Paris", entity_group="LOC", score=0.97)


def test_ml_detector_maps_labels_to_categories():
    text = "Alice lives in Paris today"
    spans = MLDetector(tagger=_fake_tagger).detect(text)
    cats = {s.category for s in spans}
    assert Category.IDENTIFIER in cats  # PER
    assert Category.PERSONAL_CONTEXT in cats  # LOC


def test_ml_detector_threshold_filters_low_score():
    def low_tagger(text: str):
        yield TaggedEntity(start=0, end=5, word="Alice", entity_group="PER", score=0.1)

    spans = MLDetector(tagger=low_tagger, score_threshold=0.5).detect("Alice lives here")
    assert spans == []


def test_ml_detector_unknown_label_is_skipped():
    def odd_tagger(text: str):
        yield TaggedEntity(start=0, end=5, word="Alice", entity_group="WEIRD", score=0.99)

    spans = MLDetector(tagger=odd_tagger).detect("Alice lives here")
    assert spans == []


def test_ml_detector_assigns_default_risk():
    def cred_tagger(text: str):
        yield TaggedEntity(start=0, end=5, word="abcde", entity_group="CREDENTIAL", score=0.9)

    spans = MLDetector(tagger=cred_tagger).detect("abcde token")
    assert spans[0].risk is RiskLevel.CRITICAL


def test_ml_detector_empty_text():
    assert MLDetector(tagger=_fake_tagger).detect("") == []


def test_ml_detector_skips_invalid_offsets():
    def bad_tagger(text: str):
        yield TaggedEntity(start=10, end=10, word="", entity_group="PER", score=0.99)

    assert MLDetector(tagger=bad_tagger).detect("anything") == []
