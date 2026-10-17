from pathlib import Path

from videotrans.task._speaker_refs import (
    clone_span_ms,
    line_uses_own_voice,
    plan_speaker_reference_spans,
    reference_cut_workers,
    source_index_for_cue,
    speaker_cap_to_nums,
    write_speaker_reference,
)


def test_each_sentence_clones_its_own_source_window():
    item = {
        "startraw": "00:07:57,060",
        "endraw": "00:07:58,610",
        "start_time_source": 477060,
        "end_time_source": 478480,
    }
    assert clone_span_ms(item, per_sentence=True) == (477060, 478480)
    assert clone_span_ms(item, per_sentence=False) is None
    assert clone_span_ms({"start_time_source": 10, "end_time_source": 10}, per_sentence=True) is None


def test_a_merged_line_does_not_shift_the_next_voice():
    sources = [
        {"start_time": 464220, "end_time": 465720, "text": "When they hear you've attacked a diplomatic"},
        {"start_time": 465720, "end_time": 467620, "text": "Do not act so surprised, your highness"},
        {"start_time": 473880, "end_time": 476760, "text": "I want to know what happened to the plans they sent you"},
        {"start_time": 477060, "end_time": 478480, "text": "I don't know what you're talking about"},
    ]
    speakers = ["spk6", "spk5", "spk5", "spk6"]
    merged = {"start_time": 464220, "end_time": 465720, "text": "Når de hører, at du har angrebet en diplomat"}
    leia = {"start_time": 477060, "end_time": 478610, "text": "Jeg ved ikke, hvad du taler om"}
    assert source_index_for_cue(merged, sources) == 0
    leia_index = source_index_for_cue(leia, sources)
    assert leia_index == 3
    assert speakers[leia_index] == "spk6"


def test_leias_scene_leads_and_the_officer_keeps_his_own_voice():
    leia = [
        (459220, 461740),
        (462300, 464000),
        (464220, 465720),
        (477060, 478480),
        (478960, 482420),
    ]
    officer = [(509160, 513840), (514460, 518000)]
    spans = leia + officer + [(0, 4000)]
    speakers = ["spk6"] * len(leia + officer) + ["spk5"]
    plan = plan_speaker_reference_spans(spans, speakers, max_ms=12000)
    assert plan["spk6"][0] == (459220, 461740)
    assert all(start < 500000 for start, _end in plan["spk6"])
    assert line_uses_own_voice(509160, 513840, plan["spk6"])
    assert not line_uses_own_voice(459220, 461740, plan["spk6"])


def test_two_speakers_get_separate_spans():
    spans = [(0, 5000), (5000, 8000), (8000, 20000), (20000, 23000)]
    speakers = ["spk0", "spk1", "spk0", "spk1"]
    plan = plan_speaker_reference_spans(spans, speakers, max_ms=12000)
    assert set(plan) == {"spk0", "spk1"}
    assert plan["spk0"][0] == (0, 5000)
    assert plan["spk1"] == [(5000, 8000)]


def test_single_speaker_keeps_per_line_clone():
    spans = [(0, 4000), (4000, 9000)]
    assert plan_speaker_reference_spans(spans, ["spk0", "spk0"]) == {}


def test_short_lines_are_skipped_when_a_longer_one_exists():
    spans = [(0, 200), (1000, 9000)]
    speakers = ["spk0", "spk0"]
    # one speaker: no shared plan
    assert plan_speaker_reference_spans(spans, speakers) == {}
    spans = [(0, 200), (1000, 9000), (10000, 11000), (12000, 18000)]
    speakers = ["spk0", "spk0", "spk1", "spk1"]
    plan = plan_speaker_reference_spans(spans, speakers, min_piece_ms=400)
    assert plan["spk0"] == [(1000, 9000)]
    assert plan["spk1"] == [(10000, 11000), (12000, 18000)]


def test_speaker_cap_matches_the_number_shown_in_the_form():
    assert speaker_cap_to_nums(0) == 0
    assert speaker_cap_to_nums(8) == 7
    assert speaker_cap_to_nums(1) == 1


def test_auto_threads_follow_the_cpu_and_stop_at_eight():
    assert reference_cut_workers(768, 12, 0) == 8
    assert reference_cut_workers(3, 12, 0) == 3
    assert reference_cut_workers(768, 12, 4) == 4
    assert reference_cut_workers(768, 12, 20) == 12


def test_one_piece_is_moved_into_the_reference(tmp_path):
    def cut(*, audio_file, ss, to, out_file):
        Path(out_file).write_bytes(b"piece")
        return True

    path = write_speaker_reference(
        speaker="spk0",
        pieces=[(0, 1000)],
        cache_folder=tmp_path,
        vocal="vocal.wav",
        cut=cut,
        concat=lambda list_file, ref_path: False,
        to_time=str,
    )
    assert Path(path).read_bytes() == b"piece"
    assert not list(tmp_path.glob("spkref-spk0-0.wav"))


def test_ten_minute_test_clip_is_a_separate_copy():
    from videotrans.util._ffmpeg_audio import test_clip_args, test_clip_target

    dest = test_clip_target("/films/Hope.mp4", "/tmp/clips", 10)
    assert dest.name == "Hope-test-10m.mp4"
    args = test_clip_args("/films/Hope.mp4", dest, 10)
    assert args[args.index("-t") + 1] == "600"
    assert args[args.index("-ss") + 1] == "0"
    assert args.index("-ss") < args.index("-i")


def test_cut_seeks_before_reading_the_file():
    from videotrans.util._ffmpeg_audio import cut_audio_args

    args = cut_audio_args(ss="00:00:01,000", to="00:00:02,000", audio_file="v.wav", out_file="o.wav")
    assert args.index("-ss") < args.index("-i")
    assert args.index("-to") < args.index("-i")


def test_reference_is_capped():
    spans = [(0, 30000), (30000, 40000)]
    speakers = ["spk0", "spk1"]
    plan = plan_speaker_reference_spans(spans, speakers, max_ms=12000)
    assert plan["spk0"] == [(0, 12000)]
    assert plan["spk1"] == [(30000, 40000)]
