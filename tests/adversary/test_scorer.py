from privategate.adversary.scorer import recovery_rate
from privategate.types import Category, RiskLevel, Span


def _span(text: str, cat: Category) -> Span:
    return Span(0, len(text), text, cat, RiskLevel.HIGH)


def test_full_recovery():
    gold = [_span("123-45-6789", Category.IDENTIFIER)]
    r = recovery_rate(["The ssn is 123-45-6789"], gold)
    assert r.overall_rate == 1.0
    assert r.n_recovered == 1


def test_no_recovery():
    gold = [_span("123-45-6789", Category.IDENTIFIER)]
    r = recovery_rate(["I have no idea"], gold)
    assert r.overall_rate == 0.0


def test_partial_recovery_per_category():
    gold = [
        _span("123-45-6789", Category.IDENTIFIER),
        _span("sk-secret", Category.CREDENTIAL),
        _span("diabetes", Category.MEDICAL),
    ]
    preds = ["123-45-6789", "no credential here", "diabetes mention"]
    r = recovery_rate(preds, gold)
    assert r.overall_rate == 2 / 3
    assert r.by_category["IDENTIFIER"]["rate"] == 1.0
    assert r.by_category["CREDENTIAL"]["rate"] == 0.0
    assert r.by_category["MEDICAL"]["rate"] == 1.0


def test_category_filter_restricts_scope():
    gold = [
        _span("123-45-6789", Category.IDENTIFIER),
        _span("diabetes", Category.MEDICAL),
    ]
    preds = ["123-45-6789"]
    r = recovery_rate(preds, gold, categories=[Category.MEDICAL])
    assert r.n_gold == 1
    assert r.overall_rate == 0.0


def test_empty_gold_returns_zero():
    r = recovery_rate(["anything"], [])
    assert r.overall_rate == 0.0
    assert r.n_gold == 0


def test_match_is_case_insensitive():
    gold = [_span("Alice Smith", Category.IDENTIFIER)]
    r = recovery_rate(["recovered ALICE smith found"], gold)
    assert r.overall_rate == 1.0
