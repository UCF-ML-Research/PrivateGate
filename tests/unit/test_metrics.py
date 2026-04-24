from __future__ import annotations

from privategate.types import Mode

from ..baselines.always_he import AlwaysHE
from ..baselines.always_plaintext import AlwaysPlaintext
from ..baselines.keyword_only import KeywordOnly
from ..baselines.regex_only import RegexOnly
from ..harness.metrics import (
    STRICTNESS,
    aggregate,
    is_downgrade,
    is_over_escalation,
    percentile,
)
from ..harness.runner import run_eval
from ..harness.schemas import GoldItem, RouterDecision


def _gi(id_: str, gold: Mode, cats: dict[str, bool] | None = None, prompt: str = "x") -> GoldItem:
    return GoldItem(
        id=id_,
        prompt=prompt,
        source="synthetic",
        categories=cats or {},
        severity={},
        spans=[],
        gold_mode=gold,
    )


def _rd(id_: str, mode: Mode, lat: float = 1.0) -> RouterDecision:
    return RouterDecision(item_id=id_, mode=mode, latency_ms=lat)


def test_strictness_order():
    assert (
        STRICTNESS[Mode.PLAINTEXT]
        < STRICTNESS[Mode.PERTURB]
        < STRICTNESS[Mode.HE_SMPC]
        < STRICTNESS[Mode.ABSTAIN]
    )


def test_is_downgrade():
    assert is_downgrade(Mode.PLAINTEXT, Mode.PERTURB)
    assert is_downgrade(Mode.PLAINTEXT, Mode.HE_SMPC)
    assert is_downgrade(Mode.PERTURB, Mode.HE_SMPC)
    assert not is_downgrade(Mode.HE_SMPC, Mode.PERTURB)
    assert not is_downgrade(Mode.PERTURB, Mode.PERTURB)


def test_is_over_escalation():
    assert is_over_escalation(Mode.HE_SMPC, Mode.PLAINTEXT)
    assert is_over_escalation(Mode.PERTURB, Mode.PLAINTEXT)
    # ABSTAIN isn't counted as over-escalation
    assert not is_over_escalation(Mode.ABSTAIN, Mode.PLAINTEXT)


def test_percentile_edges():
    assert percentile([], 50) == 0.0
    assert percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_aggregate_counts():
    items = [
        _gi("a", Mode.PERTURB, {"pii": True}),
        _gi("b", Mode.HE_SMPC, {"phi": True}),
        _gi("c", Mode.PLAINTEXT, {}),
        _gi("d", Mode.PLAINTEXT, {}),
    ]
    decisions = {
        "a": _rd("a", Mode.PLAINTEXT),    # downgrade on pii
        "b": _rd("b", Mode.PERTURB),      # downgrade on phi
        "c": _rd("c", Mode.PLAINTEXT),    # correct
        "d": _rd("d", Mode.HE_SMPC),      # over-escalation
    }
    out = aggregate("toy", items, decisions)
    assert out.n == 4
    assert abs(out.fdr_overall - 0.5) < 1e-9
    assert out.fdr_per_class["pii"] == 1.0
    assert out.fdr_per_class["phi"] == 1.0
    assert out.oer == 0.25
    assert out.abstain_rate == 0.0


def test_always_plaintext_downgrades_sensitive():
    items = [
        _gi("s1", Mode.PERTURB, {"pii": True}, "my email is a@b.com"),
        _gi("s2", Mode.HE_SMPC, {"phi": True}, "diagnosis ICD-10 I25"),
    ]
    out = run_eval(AlwaysPlaintext(), items)
    assert out.fdr_overall == 1.0


def test_always_he_over_escalates_benign():
    items = [_gi("b1", Mode.PLAINTEXT, {}, "what is the capital of France?")]
    out = run_eval(AlwaysHE(), items)
    assert out.oer == 1.0
    assert out.fdr_overall == 0.0


def test_regex_baseline_catches_email():
    items = [
        _gi("g1", Mode.PERTURB, {"pii": True}, "please email alice@example.com"),
        _gi("g2", Mode.PLAINTEXT, {}, "hello world"),
    ]
    out = run_eval(RegexOnly(), items)
    assert out.fdr_overall == 0.0


def test_keyword_baseline_catches_hipaa():
    items = [
        _gi("k1", Mode.HE_SMPC, {"phi": True}, "patient MRN 12345 with diagnosis X"),
        _gi("k2", Mode.PLAINTEXT, {}, "recommend a movie"),
    ]
    out = run_eval(KeywordOnly(), items)
    assert out.fdr_overall == 0.0
