from privategate.detector.rules import detect_dob


def test_detects_slash_dob():
    spans = detect_dob("born 03/15/1987")
    assert len(spans) == 1
    assert spans[0].text == "03/15/1987"


def test_detects_dash_dob():
    spans = detect_dob("dob 12-31-1999")
    assert len(spans) == 1


def test_no_match_for_year_only():
    assert detect_dob("the year 1987 was great") == []
