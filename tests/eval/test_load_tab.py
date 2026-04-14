from privategate.eval.datasets.tab import load_tab
from privategate.types import Category


def test_loads_fixture():
    examples = load_tab()
    assert len(examples) == 2
    assert examples[0].id == "tab-1"
    assert any(s.category is Category.IDENTIFIER for s in examples[0].spans)


def test_span_offsets_are_consistent_with_text():
    for ex in load_tab():
        for s in ex.spans:
            assert ex.text[s.start:s.end] == s.text
