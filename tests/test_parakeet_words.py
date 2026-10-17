from videotrans.process._stt_utils import _resegment
from videotrans.process.stt_pipe import group_parakeet_tokens, place_clip_words


def test_a_separate_period_stays_on_the_word():
    words = group_parakeet_tokens([
        {"token": "Ultimate", "start": 0.50, "end": 0.90},
        {"token": " evil", "start": 0.90, "end": 1.20},
        {"token": ".", "start": 1.20, "end": 1.20},
        {"token": " Make", "start": 1.30, "end": 1.55},
        {"token": " sure", "start": 1.55, "end": 1.80},
    ])
    assert [word["word"] for word in words] == ["Ultimate", "evil.", "Make", "sure"]


def test_subwords_join_until_the_next_word():
    words = group_parakeet_tokens([
        {"token": "sn", "start": 0.40, "end": 0.55},
        {"token": "akes.", "start": 0.55, "end": 0.80},
        {"token": " Yes", "start": 0.80, "end": 1.00},
    ])
    assert [word["word"] for word in words] == ["snakes.", "Yes"]


def test_the_clip_pad_is_removed_only_when_the_first_word_starts_after_it():
    padded = place_clip_words(
        [{"word": "Yes", "start": 0.50, "end": 0.80}],
        208.8,
    )
    assert padded[0]["start"] == 208.9
    assert padded[0]["end"] == 209.2
    already = place_clip_words(
        [{"word": "Yes", "start": 0.0, "end": 0.2}, {"word": "no", "start": 0.2, "end": 0.4}],
        10.0,
    )
    assert [word["start"] for word in already] == [10.0, 10.2]


def test_parakeet_words_split_at_the_period():
    words = place_clip_words(group_parakeet_tokens([
        {"token": "Ultimate", "start": 0.50, "end": 0.90},
        {"token": " evil", "start": 0.90, "end": 1.20},
        {"token": ".", "start": 1.20, "end": 1.20},
        {"token": " Make", "start": 1.25, "end": 1.55},
        {"token": " sure", "start": 1.55, "end": 1.90},
    ]), 208.8)
    lines = _resegment(
        [{"text": "Ultimate evil. Make sure", "start": 208.8, "end": 212.6, "words": words}],
        min_speech_ms=3000,
        max_speech_ms=6000,
    )
    assert [line["text"].strip() for line in lines] == ["Ultimate evil.", "Make sure"]
    assert lines[0]["end_time"] == 209600
    assert lines[1]["start_time"] == 209650
