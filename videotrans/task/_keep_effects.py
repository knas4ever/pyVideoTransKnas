"""Keep non-speech sounds that vocal separation removed with the voices.

R2-D2's beeps land in the vocal stem. Replacing that stem with a dub deletes
them. Mute the stem only where a subtitle has speech, and mix the rest back.
"""


def _time_ms(line, key):
    if isinstance(line, dict):
        return int(line.get(key) or 0)
    return int(line[key])


def speech_mute_ranges(lines, *, pad_ms=120):
    """Merged ranges, in milliseconds, where the original voice must stay out."""
    spans = []
    for line in lines or []:
        start = max(0, _time_ms(line, "start_time") - pad_ms)
        end = _time_ms(line, "end_time") + pad_ms
        if end <= start:
            continue
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def speech_mute_filter(lines, *, pad_ms=120):
    """ffmpeg volume filter that silences speech and leaves everything else."""
    ranges = speech_mute_ranges(lines, pad_ms=pad_ms)
    if not ranges:
        return None
    enabled = "+".join(
        f"between(t,{start / 1000:.3f},{end / 1000:.3f})" for start, end in ranges
    )
    return f"volume=0:enable='{enabled}'"
