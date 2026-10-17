from videotrans.task._word_times import apply_word_times, save_word_times, settle_sentence_times


def _words():
    return [
        {"word": "There", "start": 0, "end": 300},
        {"word": "goes", "start": 300, "end": 600},
        {"word": "another", "start": 600, "end": 900},
        {"word": "one", "start": 900, "end": 1200},
        {"word": "Hold", "start": 2100, "end": 2400},
        {"word": "your", "start": 2400, "end": 2600},
        {"word": "fire", "start": 2600, "end": 2900},
        {"word": "shortcircuited", "start": 4000, "end": 4600},
        {"word": "That's", "start": 4700, "end": 5000},
        {"word": "funny", "start": 5100, "end": 5500},
    ]


def test_each_sentence_uses_its_first_and_last_word():
    lines = apply_word_times(
        [
            {"text": "There goes another one", "start_time": 0, "end_time": 7760, "spk": "spk1"},
            {"text": "Hold your fire", "start_time": 7760, "end_time": 9000, "spk": "spk2"},
        ],
        _words(),
    )
    assert lines[0]["start_time"] == 0
    assert lines[0]["end_time"] == 1200
    assert lines[1]["start_time"] == 2100
    assert lines[1]["end_time"] == 2900
    assert lines[1]["startraw"] == "00:00:02,100"


def test_a_word_moved_to_the_next_sentence_takes_its_own_time():
    lines = apply_word_times(
        [
            {"text": "shortcircuited", "start_time": 4000, "end_time": 5500, "spk": "spk1"},
            {"text": "That's funny", "start_time": 5500, "end_time": 6000, "spk": "spk1"},
        ],
        _words(),
    )
    assert lines[0]["end_time"] == 4600
    assert lines[1]["start_time"] == 4700
    assert lines[1]["end_time"] == 5500


def test_an_unknown_sentence_keeps_its_time_and_the_next_sentence_still_matches():
    lines = apply_word_times(
        [
            {"text": "not spoken here", "start_time": 100, "end_time": 200, "spk": "spk0"},
            {"text": "Hold your fire", "start_time": 9000, "end_time": 9500, "spk": "spk2"},
        ],
        _words(),
    )
    assert lines[0]["start_time"] == 100
    assert lines[1]["start_time"] == 2100
    assert lines[1]["end_time"] == 2900


def test_funny_the_takes_the_voice_of_the_rest_of_the_sentence():
    words = [
        {"word": "They", "start": 431376, "end": 431456},
        {"word": "must", "start": 431456, "end": 431616},
        {"word": "have", "start": 431616, "end": 431696},
        {"word": "short", "start": 431696, "end": 431936},
        {"word": "circuited", "start": 431936, "end": 432336},
        {"word": "Funny", "start": 433376, "end": 433776},
        {"word": "the", "start": 433856, "end": 433936},
        {"word": "damage", "start": 433936, "end": 434336},
        {"word": "doesn't", "start": 434336, "end": 434496},
        {"word": "look", "start": 434496, "end": 434576},
        {"word": "as", "start": 434576, "end": 434656},
        {"word": "bad", "start": 434656, "end": 434976},
        {"word": "from", "start": 434976, "end": 435136},
        {"word": "out", "start": 435136, "end": 435296},
        {"word": "here", "start": 435296, "end": 435616},
    ]
    lines = [
        {"text": "They must have short circuited", "start_time": 431376, "end_time": 432336},
        {"text": "Funny the", "start_time": 433376, "end_time": 433936},
        # The diarization span is too long to join. The spoken words are not.
        {"text": "damage doesn't look as bad from out here", "start_time": 433936, "end_time": 438000},
    ]
    settled, speakers = settle_sentence_times(lines, ["spk16", "spk16", "spk8"], words)
    assert [line["text"] for line in settled] == [
        "They must have short circuited",
        "Funny the damage doesn't look as bad from out here",
    ]
    assert speakers == ["spk16", "spk8"]
    assert settled[1]["start_time"] == 433376
    assert settled[1]["end_time"] == 435616


def test_word_times_are_stored_as_milliseconds(tmp_path):
    log = tmp_path / "qwen.log"
    log.write_text("", encoding="utf-8")
    save_word_times(
        [{"word": "There▁", "start": 1.25, "end": 1.5}, {"word": ".", "start": 1.5, "end": 1.6}],
        log,
    )
    saved = (tmp_path / "words.json").read_text(encoding="utf-8")
    assert '"start": 1250' in saved
    assert '"end": 1500' in saved
    assert "▁" not in saved
