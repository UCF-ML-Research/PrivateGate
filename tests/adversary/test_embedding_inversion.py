from privategate.adversary.embedding_inversion import EmbeddingInverter


def test_default_recovers_input_verbatim():
    """The identity reconstructor returns the outbound text — any plaintext
    that survived the rewriter is exposed."""
    attacker = EmbeddingInverter()
    result = attacker.attack("ssn 123-45-6789")
    assert "123-45-6789" in result.predictions[0]
    assert result.attacker == "embedding_inversion"


def test_default_returns_no_predictions_for_empty_outbound():
    attacker = EmbeddingInverter()
    result = attacker.attack("")
    assert result.predictions == []


def test_custom_reconstructor_used():
    def fake(text: str):
        return ["leaked", text + " plus more"]

    result = EmbeddingInverter(reconstructor=fake).attack("hello")
    assert "leaked" in result.predictions
    assert "hello plus more" in result.predictions


def test_redacted_outbound_yields_no_pii():
    """The identity reconstructor cannot recover what was never in the outbound."""
    transformed = "my [IDENTIFIER] please help"
    result = EmbeddingInverter().attack(transformed)
    assert all("123-45-6789" not in p for p in result.predictions)
