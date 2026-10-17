from videotrans.task._clear_cache import clear_previous_run
from videotrans.task._import_srt import install_subtitle_files
from videotrans.task._stage_diariz import reuses_saved_speakers
from videotrans.task._stage_dubbing import dubbing_cache_key
from videotrans.util._srt_parse import get_subtitle_from_srt


def _srt(*rows):
    blocks = []
    for index, (start, end, text) in enumerate(rows, start=1):
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def test_a_spoken_file_replaces_the_source_and_drops_old_speakers(tmp_path):
    spoken = tmp_path / "good.srt"
    spoken.write_text(
        _srt(("00:00:01,000", "00:00:02,000", "Hello there.")),
        encoding="utf-8",
    )
    source = tmp_path / "out" / "en.srt"
    stale = tmp_path / "out" / "speaker.json"
    stale.parent.mkdir()
    stale.write_text("[]", encoding="utf-8")
    result = install_subtitle_files(
        spoken, None, source, tmp_path / "out" / "da.srt", drop_paths=(stale,)
    )
    assert result == {"spoken": 1, "target": None}
    assert get_subtitle_from_srt(source.as_posix())[0]["text"] == "Hello there."
    assert not stale.exists()


def test_a_test_clip_keeps_only_cues_inside_the_limit(tmp_path):
    spoken = tmp_path / "film.srt"
    spoken.write_text(
        _srt(
            ("00:00:10,000", "00:04:00,000", "Inside."),
            ("00:04:30,000", "00:06:00,000", "Crosses the end."),
            ("00:06:10,000", "00:06:20,000", "Later."),
        ),
        encoding="utf-8",
    )
    source = tmp_path / "en.srt"
    install_subtitle_files(spoken, None, source, tmp_path / "da.srt", limit_ms=5 * 60 * 1000)
    lines = get_subtitle_from_srt(source.as_posix())
    assert [line["text"] for line in lines] == ["Inside.", "Crosses the end."]
    assert lines[1]["end_time"] == 5 * 60 * 1000


def test_clear_empties_output_and_caches_then_a_subtitle_can_be_copied(tmp_path, monkeypatch):
    import videotrans.task._clear_cache as clear_mod

    output = tmp_path / "output"
    job = tmp_path / "job"
    voices = tmp_path / "voices"
    translations = tmp_path / "translations"
    for folder in (output, job, voices, translations):
        folder.mkdir()
        (folder / "old.txt").write_text("old", encoding="utf-8")
    kept = tmp_path / "test-clips"
    kept.mkdir()
    (kept / "clip.mkv").write_text("clip", encoding="utf-8")
    monkeypatch.setattr(clear_mod, "DUBBING_CACHE", str(voices))
    monkeypatch.setattr(clear_mod, "TRANSLATE_CACHE", str(translations))

    clear_previous_run(output, job)
    spoken = tmp_path / "spoken.srt"
    spoken.write_text(
        _srt(("00:00:01,000", "00:00:02,000", "Hello there.")),
        encoding="utf-8",
    )
    source = output / "en.srt"
    install_subtitle_files(spoken, None, source, output / "da.srt")

    assert source.is_file()
    assert list(job.iterdir()) == []
    assert list(voices.iterdir()) == []
    assert list(translations.iterdir()) == []
    assert (kept / "clip.mkv").is_file()


def test_clear_output_does_not_keep_saved_speakers():
    assert reuses_saved_speakers(clear_cache=True, speaker_json_exists=True) is False
    assert reuses_saved_speakers(clear_cache=False, speaker_json_exists=True) is True
    assert reuses_saved_speakers(clear_cache=False, speaker_json_exists=False) is False


def test_a_spoken_file_removes_the_old_translation(tmp_path):
    spoken = tmp_path / "good.srt"
    spoken.write_text(
        _srt(("00:00:01,000", "00:00:02,000", "Hello there.")),
        encoding="utf-8",
    )
    target = tmp_path / "da.srt"
    target.write_text(
        _srt(("00:00:01,000", "00:00:02,000", "Gammel tekst.")),
        encoding="utf-8",
    )
    install_subtitle_files(spoken, None, tmp_path / "en.srt", target)
    assert not target.exists()


def test_a_different_cloned_voice_gets_a_different_cache_key(tmp_path):
    first = tmp_path / "spk-a.wav"
    second = tmp_path / "spk-b.wav"
    first.write_bytes(b"voice-a")
    second.write_bytes(b"voice-b")
    shared = dict(
        text="da-Hej",
        voice="clone",
        rate="+0%",
        volume="+100%",
        pitch="+0Hz",
        tts_type=8,
        silence=(True, True, "default"),
    )
    assert dubbing_cache_key(**shared, ref_wav=first) != dubbing_cache_key(
        **shared, ref_wav=second
    )
