from privategate.detector.rules import detect_mrn
from privategate.types import Category


def test_detects_mrn_with_colon():
    spans = detect_mrn("Patient MRN: 1234567")
    assert len(spans) == 1
    assert spans[0].category is Category.MEDICAL


def test_detects_mrn_inline():
    spans = detect_mrn("mrn 9876543")
    assert len(spans) == 1


def test_no_match_random_digits():
    assert detect_mrn("7654321") == []
