from privategate.eval.detector_f1 import aggregate, compute_f1
from privategate.types import Category, RiskLevel, Span


def _s(start, end, cat=Category.IDENTIFIER):
    return Span(start, end, "x" * (end - start), cat, RiskLevel.HIGH)


def test_perfect_match_strict():
    gold = [_s(0, 5)]
    pred = [_s(0, 5)]
    r = compute_f1(pred, gold, mode="strict")
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0


def test_strict_misses_partial_overlap():
    gold = [_s(0, 5)]
    pred = [_s(0, 4)]
    r = compute_f1(pred, gold, mode="strict")
    assert r.tp == 0 and r.fp == 1 and r.fn == 1


def test_overlap_mode_credits_partial():
    gold = [_s(0, 5)]
    pred = [_s(0, 4)]
    r = compute_f1(pred, gold, mode="overlap")
    assert r.tp == 1


def test_category_mismatch_is_not_a_match():
    gold = [_s(0, 5, Category.IDENTIFIER)]
    pred = [_s(0, 5, Category.MEDICAL)]
    r = compute_f1(pred, gold, mode="overlap")
    assert r.tp == 0


def test_empty_inputs_yield_zero():
    r = compute_f1([], [], mode="strict")
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0


def test_aggregate_micro_average():
    a = compute_f1([_s(0, 5)], [_s(0, 5)], mode="strict")
    b = compute_f1([_s(0, 5)], [], mode="strict")
    agg = aggregate([a, b])
    assert agg.tp == 1 and agg.fp == 1 and agg.fn == 0
    assert 0.49 < agg.precision < 0.51
