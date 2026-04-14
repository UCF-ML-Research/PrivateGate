from privategate.eval.metrics.utility import answer_similarity, exact_match


def test_exact_match_normalizes_punctuation():
    assert exact_match("Paris.", "paris") == 1.0


def test_exact_match_mismatch():
    assert exact_match("paris", "london") == 0.0


def test_answer_similarity_identical():
    assert answer_similarity("low carb diet", "low carb diet") == 1.0


def test_answer_similarity_disjoint():
    assert answer_similarity("apple banana", "carrot daikon") == 0.0


def test_answer_similarity_partial():
    sim = answer_similarity("low carb mediterranean diet", "low carb diet")
    assert 0.0 < sim < 1.0
