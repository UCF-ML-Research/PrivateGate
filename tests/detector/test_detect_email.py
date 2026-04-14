from privategate.detector.rules import detect_email


def test_detects_email():
    spans = detect_email("contact me at jane.doe@example.com please")
    assert len(spans) == 1
    assert spans[0].text == "jane.doe@example.com"


def test_no_match_when_absent():
    assert detect_email("no email here") == []
