from videotrans.task._reference_srt import (
    format_reference,
    pair_reference,
    references_for_cues,
)


def _cue(start, end, text, stamp="00:00:00,000"):
    return {"start_time": start, "end_time": end, "text": text, "startraw": stamp}


def test_reference_pairs_follow_time_not_line_number():
    spoken = [
        _cue(0, 1000, "Hello"),
        _cue(5000, 6000, "He said to wake you up gently"),
    ]
    target = [
        _cue(5200, 7000, "Jeg ville vække dig varsomt"),
        _cue(0, 900, "Hej"),
    ]
    pairs = pair_reference(spoken, target)
    assert pairs[1][1]["text"] == "Jeg ville vække dig varsomt"


def test_a_batch_only_receives_nearby_reference_lines():
    pairs = pair_reference(
        [_cue(0, 1000, "Hello"), _cue(40000, 41000, "Wake up")],
        [_cue(0, 1000, "Hej"), _cue(40500, 42000, "Vågn")],
    )
    chosen = references_for_cues([_cue(40200, 41200, "wake up gently")], pairs)
    assert [src["text"] for src, _tgt in chosen] == ["Wake up"]


def test_one_danish_file_matches_by_time():
    danish = [(None, _cue(40500, 42000, "Jeg ville vække dig varsomt", "00:00:40,500"))]
    chosen = references_for_cues([_cue(40200, 41200, "wake up gently")], danish)
    text = format_reference(chosen)
    assert "TARGET: Jeg ville vække dig varsomt" in text
    assert "SPOKEN:" not in text


def test_one_english_file_matches_by_time():
    english = [(_cue(40200, 41200, "He said to wake you up gently", "00:00:40,200"), None)]
    chosen = references_for_cues([_cue(40000, 41000, "wake up gently")], english)
    text = format_reference(chosen)
    assert "SPOKEN: He said to wake you up gently" in text
    assert "TARGET:" not in text


def test_reference_text_does_not_include_the_dub_timestamps():
    text = format_reference([
        (_cue(1000, 2000, "Wake up", "00:40:20,080"), _cue(1000, 2000, "Vågn")),
    ])
    assert "SPOKEN: Wake up" in text
    assert "TARGET: Vågn" in text
    assert "00:40:20,080 -->" not in text
