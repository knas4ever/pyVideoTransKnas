from videotrans.tts._designations import speakable_text


def test_r2d2_is_spelled_with_hyphens():
    assert speakable_text("Hvad er det?  R2D2") == "Hvad er det?  R-2-D-2"


def test_an_existing_hyphen_is_spelled_the_same_way():
    assert speakable_text("R2-D2, hvor er du?") == "R-2-D-2, hvor er du?"
    assert speakable_text("C-3PO") == "C-3-P-O"
    assert speakable_text("BB8") == "B-B-8"


def test_ordinary_words_and_years_stay():
    assert speakable_text("Det her er vanvid i 1977.") == "Det her er vanvid i 1977."
    assert speakable_text("den R2-enhed vi købte") == "den R2-enhed vi købte"
    assert speakable_text("Hold godt fast, R2.") == "Hold godt fast, R-2."
