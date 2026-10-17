"""Add the dubbed track to the original file without re-encoding the picture."""


def keeps_original_streams(*, mux_audio, video_autorate, subtitle_type):
    """True when the finished file can be the original streams plus one audio track.

    Video slowdown changes the picture timing. Burned-in subtitles must be
    drawn onto the frames. Both of those still re-encode.
    """
    try:
        kind = int(subtitle_type or 0)
    except (TypeError, ValueError):
        kind = 0
    return bool(mux_audio) and not bool(video_autorate) and kind not in (1, 3)


def output_ext_for_mux(source, configured_ext):
    """Keep Matroska when the original film is MKV. Other sources follow the setting."""
    if str(source).lower().endswith(".mkv"):
        return ".mkv"
    return configured_ext or ".mp4"


def audio_language_tag(code, *, mkv=False):
    """ISO 639-2 tag for the generated dub, for example ``da`` -> ``dan``."""
    from videotrans.configure._languages_dict import SUBTITLE_CODE, SUBTITLE_CODE_B

    short = (code or "").replace("_", "-").split("-")[0].lower()
    tag = SUBTITLE_CODE.get(short, short or "und")
    if mkv:
        tag = SUBTITLE_CODE_B.get(tag, tag)
    return tag


def mux_command(source, audio, output, *, language, title, audio_streams):
    """ffmpeg args that copy the original picture and sound and append one dubbed track.

    An MP4 cannot hold the original SRT subtitle streams. Those stay as files
    beside the video. Matroska can keep every original stream.
    """
    index = max(0, int(audio_streams or 0))
    mp4 = str(output).lower().endswith(".mp4")
    cmd = [
        "-y",
        "-i", source,
        "-i", audio,
    ]
    if mp4:
        cmd.extend(["-map", "0:v", "-map", "0:a?", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0", "-map", "1:a:0"])
    cmd.extend([
        "-c", "copy",
        f"-metadata:s:a:{index}", f"language={language}",
        f"-metadata:s:a:{index}", f"title={title or language}",
    ])
    for i in range(index):
        cmd.extend([f"-disposition:a:{i}", "0"])
    cmd.extend([f"-disposition:a:{index}", "default"])
    if str(output).lower().endswith(".mp4"):
        cmd.extend(["-movflags", "+faststart"])
    cmd.append(output)
    return cmd
