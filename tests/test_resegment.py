from videotrans.process._stt_utils import _resegment
from videotrans.recognition._qwenasrlocal import alignment_chunk_limits


def _words(*rows):
    return [{"text": "line", "start": 0, "end": rows[-1][2], "words": [
        {"word": word, "start": start, "end": end} for word, start, end in rows
    ]}]


def test_a_pause_ends_the_sentence_before_three_seconds():
    lines = _resegment(
        _words(
            ("There", 0.0, 0.3),
            ("goes", 0.3, 0.6),
            ("another", 0.6, 0.9),
            ("one", 0.9, 1.2),
            ("Hold", 2.1, 2.4),
            ("your", 2.4, 2.6),
            ("fire", 2.6, 2.9),
        ),
        min_speech_ms=3000,
        max_speech_ms=5000,
    )
    assert [line["text"].strip() for line in lines] == [
        "There goes another one",
        "Hold your fire",
    ]
    assert lines[0]["start_time"] == 0
    assert lines[0]["end_time"] == 1200
    assert lines[1]["start_time"] == 2100
    assert lines[1]["end_time"] == 2900


def test_a_short_capital_keeps_its_own_start():
    lines = _resegment(
        _words(
            ("Hello", 0.0, 0.5),
            ("there", 0.5, 1.0),
            ("Yes", 2.0, 2.8),
        ),
        min_speech_ms=3000,
        max_speech_ms=5000,
    )
    assert [line["text"].strip() for line in lines] == ["Hello there", "Yes"]
    assert lines[1]["start_time"] == 2000
    assert lines[1]["end_time"] == 2800


def test_a_full_stop_ends_the_sentence_even_without_a_pause():
    lines = _resegment(
        _words(
            ("Jack,", 223.12, 223.40),
            ("tell", 223.42, 223.70),
            ("me", 223.70, 223.88),
            ("you", 223.88, 224.06),
            ("didn't", 224.06, 224.20),
            ("plan", 224.20, 224.44),
            ("this.", 224.44, 224.68),
            ("Are", 224.68, 224.70),
            ("you", 224.70, 224.70),
            ("serious?", 224.70, 224.78),
            ("How", 224.90, 225.08),
            ("dare", 225.08, 225.46),
            ("you?", 225.46, 225.72),
            ("Come", 227.10, 227.30),
            ("on.", 227.30, 227.58),
        ),
        min_speech_ms=3000,
        max_speech_ms=5000,
    )
    assert [line["text"].strip() for line in lines] == [
        "Jack, tell me you didn't plan this.",
        "Are you serious?",
        "How dare you?",
        "Come on.",
    ]
    assert lines[1]["start_time"] == 224680
    assert lines[3]["end_time"] == 227580


def test_a_period_hidden_by_the_segment_marker_still_ends_the_sentence():
    lines = _resegment(
        _words(
            ("I've", 212.42, 212.70),
            ("got", 212.70, 212.90),
            ("all", 212.90, 213.10),
            ("the", 213.34, 213.54),
            ("snakes.▁", 213.54, 213.90),
            ("So", 214.32, 214.50),
            ("when", 214.50, 214.80),
        ),
        min_speech_ms=3000,
        max_speech_ms=5000,
    )
    assert [line["text"].replace("▁", "").strip() for line in lines] == [
        "I've got all the snakes.",
        "So when",
    ]


def test_alignment_chunks_stay_on_pauses_and_under_twenty_seconds():
    limits = alignment_chunk_limits()
    assert limits["max_speech_duration_ms"] == 20000
    assert limits["min_speech_duration_ms"] < 60000
    assert limits["min_silent_duration_ms"] < 2000
