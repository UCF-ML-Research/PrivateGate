from privategate.backends.mock_standard import MockStandardBackend


def test_mock_standard_is_deterministic():
    b = MockStandardBackend()
    assert b.complete("hello") == b.complete("hello")
    assert "STANDARD" in b.complete("hi")
    assert "hi" in b.complete("hi")
