from privategate.detector.rules import detect_phone


def test_detects_us_phone_dashed():
    spans = detect_phone("call me at 555-867-5309 today")
    assert len(spans) == 1
    assert spans[0].text == "555-867-5309"


def test_detects_phone_with_country_code():
    spans = detect_phone("ring +1 415-555-1212 anytime")
    assert len(spans) == 1


def test_no_phone_in_plain_digits():
    assert detect_phone("the number 12345 is small") == []
