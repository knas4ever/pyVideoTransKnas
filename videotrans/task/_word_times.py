"""Keep each ASR word time and fit finished sentences to those words."""

import json
import re
from pathlib import Path

from videotrans.task._speaker_split import _ms, merge_sentence_fragments, shift_sentence_starts


def settle_sentence_times(lines, speakers, words=None):
    """Move stray words, fit each sentence to its spoken words, then join again.

    The second join sees the real word durations. A short lead such as
    ``Funny the`` can then take the voice of ``damage doesn't look...``
    instead of keeping the previous speaker.
    """
    shifted, shifted_speakers = shift_sentence_starts(lines, speakers)
    merged, merged_speakers = merge_sentence_fragments(shifted, shifted_speakers)
    if not words:
        return merged, merged_speakers
    timed = apply_word_times(merged, words)
    return merge_sentence_fragments(timed, merged_speakers)


def save_word_times(words, logs_file):
    """Write ``words.json`` next to the recognizer log. Times are milliseconds."""
    if not logs_file or not words:
        return None
    payload = []
    for word in words:
        text = str(word.get("word", "")).replace("▁", "").strip()
        if not text or not _token(text):
            continue
        payload.append({
            "word": text,
            "start": int(round(float(word["start"]) * 1000)),
            "end": int(round(float(word["end"]) * 1000)),
        })
    if not payload:
        return None
    path = Path(logs_file).parent / "words.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def apply_word_times(lines, words):
    """Set each sentence to the time of its first and last spoken word.

    Words are consumed in order. A sentence that cannot be found keeps its
    old times, and the search for the next sentence stays where it was.
    """
    cursor = 0
    output = []
    for line in lines:
        piece = dict(line.items()) if not isinstance(line, dict) else dict(line)
        tokens = _sentence_tokens(piece.get("text", ""))
        found = _match_span(words, tokens, cursor) if tokens else None
        if found is not None:
            start_at, end_at, cursor = found
            piece["start_time"] = int(words[start_at]["start"])
            piece["end_time"] = max(int(words[end_at]["end"]), piece["start_time"] + 1)
        output.append(piece)
    for index, line in enumerate(output, start=1):
        line["line"] = index
        line["startraw"] = _ms(line["start_time"])
        line["endraw"] = _ms(line["end_time"])
        line["time"] = f"{line['startraw']} --> {line['endraw']}"
    return output


def _sentence_tokens(text):
    return [token for token in (_token(part) for part in str(text).split()) if token]


def _token(text):
    return re.sub(r"[^\w']+", "", str(text).lower().replace("▁", ""))


def _match_span(words, tokens, cursor):
    """Return ``(first_index, last_index, next_cursor)`` or None."""
    if cursor >= len(words):
        return None
    start_at = _find_token(words, tokens[0], cursor, 12)
    if start_at is None:
        return None
    matched = 1
    last_at = start_at
    scan = start_at + 1
    for token in tokens[1:]:
        found = _find_token(words, token, scan, 4)
        if found is None:
            continue
        matched += 1
        last_at = found
        scan = found + 1
    needed = 1 if len(tokens) == 1 else max(2, (len(tokens) + 1) // 2)
    if matched < needed:
        return None
    return start_at, last_at, last_at + 1


def _find_token(words, token, cursor, window):
    limit = min(len(words), cursor + window)
    for index in range(cursor, limit):
        if _token(words[index].get("word", "")) == token:
            return index
        if index + 1 < limit:
            joined = _token(words[index].get("word", "")) + _token(words[index + 1].get("word", ""))
            if joined == token:
                return index + 1
    return None
