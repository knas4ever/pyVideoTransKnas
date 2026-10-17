from videotrans.task._mux import audio_language_tag, keeps_original_streams, mux_command, output_ext_for_mux


def test_an_mkv_source_stays_mkv():
    assert output_ext_for_mux("/film.mkv", ".mp4") == ".mkv"
    assert output_ext_for_mux("/film.MKV", ".mp4") == ".mkv"
    assert output_ext_for_mux("/film.mp4", ".mp4") == ".mp4"


def test_danish_track_tag():
    assert audio_language_tag("da") == "dan"
    assert audio_language_tag("en-US") == "eng"


def test_mux_is_skipped_when_the_picture_must_change():
    assert keeps_original_streams(mux_audio=True, video_autorate=False, subtitle_type=0)
    assert not keeps_original_streams(mux_audio=False, video_autorate=False, subtitle_type=0)
    assert not keeps_original_streams(mux_audio=True, video_autorate=True, subtitle_type=0)
    assert not keeps_original_streams(mux_audio=True, video_autorate=False, subtitle_type=1)


def test_mux_command_keeps_the_original_and_tags_the_new_track():
    cmd = mux_command(
        "/film.mkv",
        "/dub.m4a",
        "/out.mp4",
        language="dan",
        title="Danish",
        audio_streams=1,
    )
    assert "0:v" in cmd and "0:a?" in cmd
    assert "1:a:0" in cmd
    mkv = mux_command(
        "/film.mkv", "/dub.m4a", "/out.mkv",
        language="dan", title="Danish", audio_streams=1,
    )
    assert mkv[mkv.index("-map") + 1] == "0"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert "language=dan" in cmd
    assert "title=Danish" in cmd
    assert cmd[cmd.index("-disposition:a:0") + 1] == "0"
    assert cmd[cmd.index("-disposition:a:1") + 1] == "default"
