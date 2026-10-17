"""Song lines marked by the translator stay in the timeline but are not dubbed."""


def is_song_marker(text):
    token = (text or "").strip().strip(".").upper()
    return token in {"[SONG]", "SONG"}


def without_song_cues(source, target):
    """Drop pairs whose translation is exactly the song marker.

    Counts must match. A missing translation is not treated as a song, so a
    short model answer cannot delete dialogue.
    """
    if len(source) != len(target):
        return source, target, []
    keep_source = []
    keep_target = []
    dropped_at = []
    for index, (src, tgt) in enumerate(zip(source, target)):
        if is_song_marker(tgt.get("text")):
            dropped_at.append(index)
            continue
        keep_source.append(src)
        keep_target.append(tgt)
    if not keep_source:
        return source, target, []
    return keep_source, keep_target, dropped_at
