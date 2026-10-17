"""Match an uploaded subtitle pair to the cues being translated.

The uploaded files are not the dub script. They only suggest wording for
lines that happen within a few seconds of each other.
"""

REFERENCE_WINDOW_MS = 4000


def _overlap(start_a, end_a, start_b, end_b):
    return min(end_a, end_b) - max(start_a, start_b)


def pair_reference(spoken, target, *, window_ms=REFERENCE_WINDOW_MS):
    """Pair each spoken reference line with the target line nearest in time."""
    pairs = []
    for src in spoken:
        src_start = int(src["start_time"])
        src_end = int(src["end_time"])
        best = None
        best_score = None
        for tgt in target:
            tgt_start = int(tgt["start_time"])
            tgt_end = int(tgt["end_time"])
            overlap = _overlap(src_start, src_end, tgt_start, tgt_end)
            delta = abs(src_start - tgt_start)
            if overlap <= 0 and delta > window_ms:
                continue
            score = (overlap, -delta)
            if best_score is None or score > best_score:
                best_score = score
                best = tgt
        if best is not None:
            pairs.append((src, best))
    return pairs


def references_for_cues(cues, pairs, *, window_ms=REFERENCE_WINDOW_MS):
    """Reference pairs that sit within ``window_ms`` of a cue in this batch."""
    chosen = []
    seen = set()
    for cue in cues:
        cue_start = int(cue["start_time"])
        cue_end = int(cue["end_time"])
        for index, (src, tgt) in enumerate(pairs):
            if index in seen:
                continue
            anchor = src if src is not None else tgt
            anchor_start = int(anchor["start_time"])
            anchor_end = int(anchor["end_time"])
            overlap = _overlap(cue_start, cue_end, anchor_start, anchor_end)
            if overlap > 0 or abs(cue_start - anchor_start) <= window_ms:
                seen.add(index)
                chosen.append((src, tgt))
    return chosen


def format_reference(pairs):
    if not pairs:
        return ""
    has_target = any(tgt is not None for _src, tgt in pairs)
    intro = (
        "Existing translation of the same moment. Use the names and wording."
        if has_target
        else "Original wording of the same moment. Use the names."
    )
    lines = [
        intro,
        "Do not output these lines. Do not copy their timestamps.",
    ]
    for src, tgt in pairs:
        stamp = (src or tgt or {}).get("startraw") or ""
        if stamp:
            lines.append(stamp)
        if src is not None:
            lines.append("SPOKEN: " + " ".join((src.get("text") or "").split()))
        if tgt is not None:
            lines.append("TARGET: " + " ".join((tgt.get("text") or "").split()))
    return "\n".join(lines)


def load_reference_uploads(spoken_upload, target_upload, *, window_ms=REFERENCE_WINDOW_MS):
    """Return reference lines from one or both uploads, otherwise None.

    One file is matched to the cue by time. Two files are paired with each
    other first. The uploads are not installed as the dub subtitles.
    """
    from videotrans.task._import_srt import _upload_path
    from videotrans.util.help_srt import get_subtitle_from_srt

    spoken_path = _upload_path(spoken_upload)
    target_path = _upload_path(target_upload)
    if spoken_path is None and target_path is None:
        return None
    spoken = get_subtitle_from_srt(spoken_path.as_posix(), is_file=True) if spoken_path else []
    target = get_subtitle_from_srt(target_path.as_posix(), is_file=True) if target_path else []
    if spoken and target:
        return pair_reference(spoken, target, window_ms=window_ms)
    if target:
        return [(None, line) for line in target]
    if spoken:
        return [(line, None) for line in spoken]
    return None
