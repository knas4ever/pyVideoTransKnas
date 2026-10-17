from videotrans.task._speaker_split import (
    cluster_by_cosine,
    merge_sentence_fragments,
    shift_sentence_starts,
    split_runon_sentences,
    split_subtitle_lines,
    split_text_for_durations,
)


def test_different_embeddings_become_two_speakers():
    labels = cluster_by_cosine([[1, 0], [0.2, 0.98], [0.99, 0.05]], min_cosine=0.55)
    assert labels == [0, 1, 0]


def test_capital_word_marks_the_second_speaker():
    text = "It's a space station It's too big to be a space station"
    assert split_text_for_durations(text, [1380, 1210]) == [
        "It's a space station",
        "It's too big to be a space station",
    ]


class _SrtItem:
    """Same shape as ``videotrans.task.taskcfg.SrtItem``: item access plus
    ``items()``, while ``dict(item)`` fails because iteration yields names.
    """

    def __init__(self, **fields):
        self._fields = fields

    def __getitem__(self, key):
        return self._fields[key]

    def get(self, key, default=None):
        return self._fields.get(key, default)

    def items(self):
        return self._fields.items()

    def __iter__(self):
        return iter(self._fields)


def test_srt_item_splits_without_dict_cast():
    line = _SrtItem(
        line=1,
        text="It's a space station It's too big to be a space station",
        start_time=800,
        end_time=3520,
        startraw="00:00:00,800",
        endraw="00:00:03,520",
        time="00:00:00,800 --> 00:00:03,520",
        spk="",
    )
    spans = [
        {"start": 790, "end": 2170, "speaker": "spk0"},
        {"start": 2280, "end": 3490, "speaker": "spk1"},
    ]
    new_lines, speakers = split_subtitle_lines([line], spans)
    assert speakers == ["spk0", "spk1"]
    assert new_lines[0]["text"] == "It's a space station"


def test_a_question_cut_on_breaths_becomes_one_line():
    lines = [
        {"line": 29, "text": "here Are", "start_time": 436030, "end_time": 436654},
        {"line": 30, "text": "you sure", "start_time": 437414, "end_time": 437819},
        {"line": 31, "text": "this thing is safe", "start_time": 438325, "end_time": 439568},
    ]
    merged, speakers = merge_sentence_fragments(lines, ["spk23", "spk24", "spk8"])
    assert len(merged) == 1
    assert merged[0]["text"] == "here Are you sure this thing is safe"
    assert merged[0]["start_time"] == 436030
    assert merged[0]["end_time"] == 439568
    assert speakers == ["spk8"]


def test_the_first_word_of_the_next_sentence_moves_forward():
    lines = [
        {
            "line": 1,
            "text": "It must have shortcircuited That's",
            "start_time": 428000,
            "end_time": 432288,
        },
        {"line": 2, "text": "funny The", "start_time": 433488, "end_time": 434039},
    ]
    shifted, _speakers = shift_sentence_starts(lines, ["spk16", "spk16"])
    assert shifted[0]["text"] == "It must have shortcircuited"
    assert shifted[1]["text"] == "That's funny The"
    assert shifted[0]["end_time"] == 432288
    assert shifted[1]["start_time"] == 433488


def test_a_capital_name_stays_on_its_own_line():
    lines = [
        {"line": 1, "text": "Where are you going The Death Star", "start_time": 0, "end_time": 4000},
        {"line": 2, "text": "plans are not in the main computer", "start_time": 4500, "end_time": 8000},
    ]
    shifted, _speakers = shift_sentence_starts(lines, ["spk8", "spk3"])
    assert shifted[0]["text"].endswith("Star")
    assert shifted[1]["text"].startswith("plans")


def test_hold_your_fire_is_not_glued_to_the_previous_sentence():
    lines = [{
        "line": 18,
        "text": "There goes another one Hold your fire There's no life force It must have shortcircuited That's",
        "start_time": 420688,
        "end_time": 432288,
    }]
    split_lines, speakers = split_runon_sentences(lines, ["spk16"])
    texts = [line["text"] for line in split_lines]
    assert texts[0] == "There goes another one"
    assert "Hold your fire" in texts
    assert texts[-1] == "That's"
    assert split_lines[0]["start_time"] == 420688
    assert split_lines[-1]["end_time"] == 432288
    assert speakers == ["spk16"] * len(texts)
    funny = {
        "line": 19,
        "text": "funny The",
        "start_time": 433488,
        "end_time": 434039,
    }
    merged, merged_speakers = merge_sentence_fragments(split_lines + [funny], speakers + ["spk16"])
    assert merged[-1]["text"] == "That's funny The"
    assert merged_speakers[-1] == "spk16"
    assert all(line["text"] != texts[0] or "Hold" not in line["text"] for line in merged)


def test_another_persons_capitalised_line_stays_separate():
    lines = [
        {
            "line": 1,
            "text": "damage doesn't look as bad from out here",
            "start_time": 433583,
            "end_time": 436000,
        },
        {
            "line": 2,
            "text": "Are you sure this thing is safe",
            "start_time": 436400,
            "end_time": 439568,
        },
    ]
    merged, speakers = merge_sentence_fragments(lines, ["spk16", "spk8"])
    assert [line["text"] for line in merged] == [
        "damage doesn't look as bad from out here",
        "Are you sure this thing is safe",
    ]
    assert speakers == ["spk16", "spk8"]


def test_the_same_speaker_is_joined_across_a_breath():
    lines = [
        {"line": 1, "text": "There'll be no escape", "start_time": 1000, "end_time": 2500},
        {"line": 2, "text": "for the princess this time", "start_time": 3100, "end_time": 4800},
    ]
    merged, speakers = merge_sentence_fragments(lines, ["spk2", "spk2"])
    assert len(merged) == 1
    assert merged[0]["text"] == "There'll be no escape for the princess this time"
    assert speakers == ["spk2"]


def test_a_finished_sentence_stays_separate():
    lines = [
        {"line": 1, "text": "This is madness.", "start_time": 0, "end_time": 800},
        {"line": 2, "text": "We're doomed.", "start_time": 1000, "end_time": 1800},
    ]
    merged, speakers = merge_sentence_fragments(lines, ["spk0", "spk1"])
    assert len(merged) == 2
    assert speakers == ["spk0", "spk1"]


def test_each_word_stays_with_the_speaker_who_said_it():
    text = (
        "The Death Star plans are not in the main computer "
        "where those transmissions you intercept what have you done with"
    )
    times = [
        (341440, 341960, "The"), (341960, 342220, "Death"), (342220, 342500, "Star"),
        (342500, 342940, "plans"), (342940, 343200, "are"), (343200, 343580, "not"),
        (343580, 343740, "in"), (343740, 343820, "the"), (343820, 343980, "main"),
        (343980, 344500, "computer"), (344500, 345360, "where"), (345360, 345680, "those"),
        (345680, 346540, "transmissions"), (346540, 346820, "you"), (346820, 347300, "intercept"),
        (347300, 348000, "what"), (348000, 348440, "have"), (348440, 348620, "you"),
        (348620, 349040, "done"), (349040, 349280, "with"),
    ]
    line = {
        "line": 14,
        "text": text,
        "start_time": 341440,
        "end_time": 349280,
    }
    spans = [
        {"start": 341783, "end": 344652, "speaker": "spk4"},
        {"start": 345107, "end": 350237, "speaker": "spk5"},
    ]
    words = [{"word": word, "start": start, "end": end} for start, end, word in times]
    new_lines, speakers = split_subtitle_lines([line], spans, words=words)
    assert speakers == ["spk4", "spk5"]
    assert new_lines[0]["text"] == "The Death Star plans are not in the main computer"
    assert new_lines[1]["text"].startswith("where those transmissions")
    assert new_lines[0]["end_time"] == 344500
    assert new_lines[1]["start_time"] == 344500


def test_one_subtitle_splits_at_the_speaker_change():
    lines = [{
        "line": 1,
        "text": "It's a space station It's too big to be a space station",
        "start_time": 800,
        "end_time": 3520,
        "startraw": "00:00:00,800",
        "endraw": "00:00:03,520",
        "time": "00:00:00,800 --> 00:00:03,520",
    }]
    spans = [
        {"start": 790, "end": 2170, "speaker": "spk0"},
        {"start": 2280, "end": 3490, "speaker": "spk1"},
    ]
    new_lines, speakers = split_subtitle_lines(lines, spans)
    assert speakers == ["spk0", "spk1"]
    assert new_lines[0]["text"] == "It's a space station"
    assert new_lines[1]["text"] == "It's too big to be a space station"
    assert new_lines[0]["start_time"] == 800
    assert new_lines[1]["end_time"] == 3490
