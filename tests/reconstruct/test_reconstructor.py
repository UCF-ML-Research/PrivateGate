from privategate.reconstruct.reconstructor import Reconstructor


def test_full_reconstruction():
    r = Reconstructor()
    out = r.reconstruct("answer = [SLOT_AB0]", {"[SLOT_AB0]": "42"})
    assert out == "answer = 42"


def test_idempotent_when_nothing_to_replace():
    r = Reconstructor()
    assert r.reconstruct("nothing here", {}) == "nothing here"
