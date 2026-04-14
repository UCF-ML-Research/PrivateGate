from privategate.adversary.llm_guesser import LLMGuesser, naive_pattern_model


def test_naive_model_extracts_digit_groups():
    guesses = naive_pattern_model("TRANSFORMED:my ssn is 123-45-6789 here")
    assert "123-45-6789" in guesses


def test_naive_model_extracts_email():
    guesses = naive_pattern_model("TRANSFORMED:contact me at jane@example.com")
    assert "jane@example.com" in guesses


def test_naive_model_extracts_credentials():
    guesses = naive_pattern_model("TRANSFORMED:export sk-abcdefgh12345678")
    assert "sk-abcdefgh12345678" in guesses


def test_naive_model_finds_nothing_in_redacted_text():
    guesses = naive_pattern_model("TRANSFORMED:my [IDENTIFIER] and [CREDENTIAL]")
    assert guesses.strip() == ""


def test_guesser_uses_injected_model():
    def fake(prompt: str) -> str:
        return "first guess\nsecond guess\n\n"

    result = LLMGuesser(model_fn=fake).attack("anything")
    assert result.predictions == ["first guess", "second guess"]


def test_guesser_recovers_leaked_ssn_via_default_model():
    result = LLMGuesser().attack("my ssn is 123-45-6789 please")
    assert any("123-45-6789" in p for p in result.predictions)


def test_guesser_recovers_nothing_from_clean_redaction():
    result = LLMGuesser().attack("my [IDENTIFIER] please help")
    assert result.predictions == []
