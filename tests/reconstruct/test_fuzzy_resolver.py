from privategate.reconstruct.fuzzy_resolver import (
    UNRESOLVED_MARKER,
    resolve_paraphrased,
)


def test_unknown_placeholder_marked():
    out = resolve_paraphrased("see [SLOT_ZZ9]", {})
    assert UNRESOLVED_MARKER in out


def test_known_placeholder_resolved():
    out = resolve_paraphrased("see [SLOT_ZZ9]", {"[SLOT_ZZ9]": "secret"})
    assert out == "see secret"


def test_no_placeholders_unchanged():
    assert resolve_paraphrased("hello world", {}) == "hello world"
