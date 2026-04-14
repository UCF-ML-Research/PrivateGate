import math

from privategate.router.divergence import answer_divergence


def test_identical_strings_zero():
    assert answer_divergence("hello world", "hello world") == 0.0


def test_disjoint_strings_one():
    assert answer_divergence("apple banana", "carrot daikon") == 1.0


def test_partial_overlap_in_unit_interval():
    d = answer_divergence("the quick brown fox", "the slow brown dog")
    assert 0.0 < d < 1.0


def test_case_and_punctuation_insensitive():
    d = answer_divergence("Hello, World!", "hello world")
    assert d == 0.0


def test_empty_inputs_are_identical():
    assert answer_divergence("", "") == 0.0


def test_custom_embedder_used():
    def emb(text: str):
        return [1.0, 0.0] if text == "a" else [0.0, 1.0]

    d = answer_divergence("a", "b", embedder=emb)
    assert math.isclose(d, 0.5, abs_tol=1e-9)


def test_custom_embedder_identical_vectors_zero():
    def emb(text: str):
        return [1.0, 1.0]

    assert math.isclose(answer_divergence("x", "y", embedder=emb), 0.0, abs_tol=1e-9)
