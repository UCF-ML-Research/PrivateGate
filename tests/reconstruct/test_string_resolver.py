from privategate.reconstruct.string_resolver import resolve_verbatim


def test_replaces_token():
    out = resolve_verbatim("answer is [SLOT_AB0]", {"[SLOT_AB0]": "secret"})
    assert out == "answer is secret"


def test_no_change_without_token():
    assert resolve_verbatim("plain", {}) == "plain"
