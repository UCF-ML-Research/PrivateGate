from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed
from privategate.types import Category


def test_seed_loads_at_least_five_examples():
    examples = load_synthetic_mixed()
    assert len(examples) >= 5


def test_seed_has_credential_and_clean_examples():
    examples = load_synthetic_mixed()
    cats = {s.category for ex in examples for s in ex.spans}
    assert Category.CREDENTIAL in cats
    assert any(len(ex.spans) == 0 for ex in examples)


def test_span_offsets_consistent():
    for ex in load_synthetic_mixed():
        for s in ex.spans:
            assert ex.text[s.start:s.end] == s.text, f"mismatch on {ex.id}: {s}"
