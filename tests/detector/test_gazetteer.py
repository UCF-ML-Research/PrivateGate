from privategate.detector.gazetteer import GazetteerDetector
from privategate.types import Category


def test_detects_medical_term():
    d = GazetteerDetector()
    spans = d.detect("I have type 2 diabetes and need help")
    assert any(s.category is Category.MEDICAL and "diabetes" in s.text.lower() for s in spans)


def test_prefers_longest_medical_term():
    d = GazetteerDetector()
    spans = d.detect("metastatic pancreatic cancer is serious")
    medical = [s for s in spans if s.category is Category.MEDICAL]
    assert len(medical) == 1
    assert medical[0].text.lower() == "metastatic pancreatic cancer"


def test_detects_titled_name():
    d = GazetteerDetector()
    spans = d.detect("met with Dr. Alice Smith yesterday")
    assert any(s.category is Category.IDENTIFIER for s in spans)
