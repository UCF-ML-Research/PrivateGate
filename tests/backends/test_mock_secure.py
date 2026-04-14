from privategate.backends.mock_secure import MockSecureBackend


def test_mock_secure_is_deterministic():
    b = MockSecureBackend()
    assert b.complete("hello") == b.complete("hello")
    assert "SECURE" in b.complete("x")
