"""Split one subtitle line when a short clip hides a second speaker."""

import re


def cluster_by_cosine(vectors, *, min_cosine=0.55):
    """Greedy clustering. ``min_cosine`` is the lowest similarity that stays one speaker."""
    labels = []
    centroids = []
    for vector in vectors:
        if not centroids:
            centroids.append(list(vector))
            labels.append(0)
            continue
        scores = [_cosine(vector, centroid) for centroid in centroids]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= min_cosine:
            labels.append(best)
            centroids[best] = _mean(centroids[best], vector)
        else:
            labels.append(len(centroids))
            centroids.append(list(vector))
    return labels


def split_text_for_durations(text, durations):
    """Split ``text`` into ``len(durations)`` pieces when no word times exist.

    An internal capitalised word is treated as the next speaker's first word.
    Otherwise the split follows the duration of each piece.
    """
    words = text.split()
    count = len(durations)
    if count <= 1 or len(words) < count:
        return [text] + [""] * (count - 1)
    if count == 2:
        capitals = [i for i, word in enumerate(words) if i and word[:1].isupper()]
        if len(capitals) == 1 and 0 < capitals[0] < len(words):
            cut = capitals[0]
            return [" ".join(words[:cut]), " ".join(words[cut:])]
    total = sum(max(int(item), 1) for item in durations)
    pieces = []
    cursor = 0
    used = 0
    for index, duration in enumerate(durations):
        if index == count - 1:
            pieces.append(" ".join(words[cursor:]))
            break
        used += max(int(duration), 1)
        end = max(cursor + 1, min(len(words) - (count - index - 1), round(len(words) * used / total)))
        pieces.append(" ".join(words[cursor:end]))
        cursor = end
    return pieces


_SENTENCE_END = set(".?!。？！")


def _ends_sentence(text):
    text = (text or "").strip()
    return bool(text) and text[-1] in _SENTENCE_END


def _starts_new_turn(text):
    text = (text or "").strip()
    return bool(text) and text[:1].isupper()


def _sentence_parts(text):
    """Split a run-on cue where a new sentence starts with a capital."""
    words = (text or "").split()
    if len(words) < 4:
        return [text] if text else []
    parts = []
    buffer = [words[0]]
    for word in words[1:]:
        if word[:1].isupper() and buffer[-1][:1].islower():
            parts.append(" ".join(buffer))
            buffer = [word]
        else:
            buffer.append(word)
    if buffer:
        parts.append(" ".join(buffer))
    return parts


def split_runon_sentences(lines, speakers=None, *, min_duration_ms=5000):
    """Give each sentence in a long cue its own time.

    ``There goes another one Hold your fire There's no life force`` is two
    people's lines in one 11-second cue. The capital letters mark the breaks,
    and the time is shared by the number of words.
    """
    if not lines:
        return [], []
    speakers = list(speakers or [])
    output = []
    output_speakers = []
    for index, line in enumerate(lines):
        piece = _copy_line(line)
        speaker = speakers[index] if index < len(speakers) else (piece.get("spk") or "spk0")
        start = int(piece["start_time"])
        end = int(piece["end_time"])
        parts = _sentence_parts(piece.get("text", ""))
        if end - start < min_duration_ms or len(parts) < 2:
            output.append(piece)
            output_speakers.append(speaker)
            continue
        weights = [max(len(part.split()), 1) for part in parts]
        total = sum(weights)
        cursor = start
        duration = end - start
        for part_index, (part, weight) in enumerate(zip(parts, weights)):
            if part_index == len(parts) - 1:
                part_end = end
            else:
                part_end = cursor + max(int(duration * weight / total), 200)
                part_end = min(part_end, end - 200 * (len(parts) - part_index - 1))
            item = dict(piece)
            item["text"] = part
            item["start_time"] = cursor
            item["end_time"] = part_end
            output.append(item)
            output_speakers.append(speaker)
            cursor = part_end
    for index, line in enumerate(output, start=1):
        line["line"] = index
        line["startraw"] = _ms(line["start_time"])
        line["endraw"] = _ms(line["end_time"])
        line["time"] = f"{line['startraw']} --> {line['endraw']}"
        line["spk"] = output_speakers[index - 1]
    return output, output_speakers


def _last_word_belongs_to_next(previous_text, next_text):
    """The first word of the next sentence was left on the line before it.

    ``shortcircuited That's`` / ``funny`` becomes ``shortcircuited`` / ``That's funny``.
    A capital name such as ``Death Star`` stays put, because both words are capital.
    """
    previous_words = (previous_text or "").split()
    next_words = (next_text or "").split()
    if len(previous_words) < 2 or not next_words:
        return False
    last_word = previous_words[-1]
    before = previous_words[-2]
    if not last_word[:1].isupper() or before[:1].isupper():
        return False
    return not next_words[0][:1].isupper()


def shift_sentence_starts(lines, speakers=None, *, max_gap_ms=1500):
    """Move a stranded sentence-opening word onto the following line."""
    if not lines:
        return [], []
    speakers = list(speakers or [])
    output = [_copy_line(line) for line in lines]
    output_speakers = [
        speakers[index] if index < len(speakers) else (line.get("spk") or "spk0")
        for index, line in enumerate(output)
    ]
    for index in range(len(output) - 1):
        current = output[index]
        nxt = output[index + 1]
        gap = int(nxt["start_time"]) - int(current["end_time"])
        if gap > max_gap_ms or not _last_word_belongs_to_next(current.get("text", ""), nxt.get("text", "")):
            continue
        words = current["text"].split()
        moved = words[-1]
        current["text"] = " ".join(words[:-1])
        nxt["text"] = f"{moved} {nxt['text'].lstrip()}".strip()
    for index, line in enumerate(output, start=1):
        line["line"] = index
        line["startraw"] = _ms(line["start_time"])
        line["endraw"] = _ms(line["end_time"])
        line["time"] = f"{line['startraw']} --> {line['endraw']}"
        line["spk"] = output_speakers[index - 1]
    return output, output_speakers


def merge_sentence_fragments(
    lines,
    speakers=None,
    *,
    max_gap_ms=1200,
    max_total_ms=12000,
    max_other_speaker_ms=2500,
    max_other_speaker_gap_ms=800,
):
    """Join one speaker's sentence when a breath cut it in two.

    A different speaker is joined only when the next cue is a short lowercase
    continuation, such as ``you sure`` after ``here Are``. A capitalised line
    from someone else stays separate.
    """
    if not lines:
        return [], []
    speakers = list(speakers or [])
    merged = []
    merged_speakers = []
    owned_ms = []
    for index, line in enumerate(lines):
        piece = _copy_line(line)
        speaker = speakers[index] if index < len(speakers) else (piece.get("spk") or "spk0")
        piece_ms = int(piece["end_time"]) - int(piece["start_time"])
        if not merged:
            merged.append(piece)
            merged_speakers.append(speaker)
            owned_ms.append(piece_ms)
            continue
        previous = merged[-1]
        gap = int(piece["start_time"]) - int(previous["end_time"])
        total_ms = int(piece["end_time"]) - int(previous["start_time"])
        same_speaker = speaker == merged_speakers[-1]
        same_speaker_continuation = same_speaker and not _starts_new_turn(piece.get("text", ""))
        other_speaker_fragment = (
            not same_speaker
            and not _starts_new_turn(piece.get("text", ""))
            and piece_ms <= max_other_speaker_ms
            and gap <= max_other_speaker_gap_ms
        )
        if (
            not _ends_sentence(previous.get("text", ""))
            and -200 <= gap <= (max_gap_ms if same_speaker else max_other_speaker_gap_ms)
            and 0 < total_ms <= max_total_ms
            and (same_speaker_continuation or other_speaker_fragment)
        ):
            if piece_ms > owned_ms[-1]:
                merged_speakers[-1] = speaker
                owned_ms[-1] = piece_ms
            previous["text"] = (
                f"{previous.get('text', '').rstrip()} {piece.get('text', '').lstrip()}"
            ).strip()
            previous["end_time"] = max(int(previous["end_time"]), int(piece["end_time"]))
            continue
        merged.append(piece)
        merged_speakers.append(speaker)
        owned_ms.append(piece_ms)
    for index, line in enumerate(merged, start=1):
        line["line"] = index
        line["startraw"] = _ms(line["start_time"])
        line["endraw"] = _ms(line["end_time"])
        line["time"] = f"{line['startraw']} --> {line['endraw']}"
        line["spk"] = merged_speakers[index - 1]
    return merged, merged_speakers


def split_subtitle_lines(lines, spans, *, min_overlap_ms=350, words=None):
    """Return ``(new_lines, speakers)``.

    A line that covers two speaker spans becomes one line per speaker.
    ``spans`` are ``{"start", "end", "speaker"}`` in milliseconds. When
    ``words`` are present, each word stays with the speaker who was talking
    at that word. Duration guessing is only used when the words cannot be
    aligned.
    """
    ordered = sorted(spans, key=lambda item: (int(item["start"]), int(item["end"])))
    new_lines = []
    speakers = []
    cursor = 0
    for line in lines:
        covered = _covered_spans(line, ordered, min_overlap_ms)
        pairs = None
        if words and len({item["speaker"] for item in covered}) >= 2:
            pairs, cursor = _align_line_words(line, words, cursor)
        elif words:
            cursor = _cursor_after_line(words, cursor, int(line["end_time"]))
        if pairs:
            groups = _groups_by_speaker(pairs, ordered)
        else:
            groups = []
        if len(groups) < 2:
            # Aligned words that all belong to one person stay one line.
            # Guessing is only for a line whose words could not be aligned.
            if pairs or len({item["speaker"] for item in covered}) < 2:
                speaker = (
                    groups[0]["speaker"] if groups
                    else covered[0]["speaker"] if covered
                    else (line.get("spk") or "spk0")
                )
                new_lines.append(_copy_line(line))
                speakers.append(speaker)
                continue
            durations = [item["end"] - item["start"] for item in covered]
            parts = split_text_for_durations(line.get("text", ""), durations)
            for item, text in zip(covered, parts):
                piece = _copy_line(line)
                piece["text"] = text
                piece["start_time"] = item["start"]
                piece["end_time"] = item["end"]
                new_lines.append(piece)
                speakers.append(item["speaker"])
            continue
        for group in groups:
            piece = _copy_line(line)
            piece["text"] = " ".join(group["tokens"])
            piece["start_time"] = group["start"]
            piece["end_time"] = group["end"]
            new_lines.append(piece)
            speakers.append(group["speaker"])
    for index, line in enumerate(new_lines, start=1):
        line["line"] = index
        line["startraw"] = _ms(line["start_time"])
        line["endraw"] = _ms(line["end_time"])
        line["time"] = f"{line['startraw']} --> {line['endraw']}"
    return new_lines, speakers


def _token(text):
    return re.sub(r"[^\w']+", "", str(text).lower().replace("▁", ""))


def _align_line_words(line, words, cursor):
    """Pair each word of the line with its timed word, or return no pairs."""
    tokens = [token for token in str(line.get("text", "")).split() if _token(token)]
    if not tokens:
        return None, cursor
    start = int(line["start_time"])
    end = int(line["end_time"])
    first = _find_word(words, tokens[0], cursor, start, end, window=24)
    if first is None:
        return None, _cursor_after_line(words, cursor, end)
    pairs = [(tokens[0], words[first])]
    scan = first + 1
    for token in tokens[1:]:
        found = _find_word(words, token, scan, start, end, window=6)
        if found is None:
            return None, _cursor_after_line(words, cursor, end)
        pairs.append((token, words[found]))
        scan = found + 1
    return pairs, scan


def _find_word(words, token, cursor, line_start, line_end, window):
    want = _token(token)
    if not want:
        return None
    limit = min(len(words), cursor + window)
    for index in range(cursor, limit):
        word = words[index]
        if _token(word.get("word", "")) != want:
            continue
        start = int(word["start"])
        end = int(word["end"])
        if end < line_start - 400 or start > line_end + 400:
            continue
        return index
    return None


def _cursor_after_line(words, cursor, line_end):
    index = cursor
    while index < len(words) and int(words[index]["start"]) < line_end:
        index += 1
    return index


def _groups_by_speaker(pairs, spans):
    groups = []
    for token, word in pairs:
        start = int(word["start"])
        end = int(word["end"])
        span = _speaker_at(start, end, spans)
        if span is None:
            return []
        speaker = span["speaker"]
        if groups and groups[-1]["speaker"] == speaker:
            groups[-1]["tokens"].append(token)
            groups[-1]["end"] = max(groups[-1]["end"], end)
            continue
        groups.append({
            "speaker": speaker,
            "tokens": [token],
            "start": start,
            "end": end,
        })
    return groups


def _speaker_at(start, end, spans):
    """The span that contains the word, or the nearest span when it sits in a gap."""
    if not spans:
        return None
    midpoint = (int(start) + int(end)) / 2
    containing = [
        span for span in spans
        if int(span["start"]) <= midpoint <= int(span["end"])
    ]
    if containing:
        return min(containing, key=lambda span: int(span["end"]) - int(span["start"]))
    return min(spans, key=lambda span: _distance(midpoint, span))


def _distance(midpoint, span):
    if int(span["start"]) <= midpoint <= int(span["end"]):
        return 0
    return min(abs(midpoint - int(span["start"])), abs(midpoint - int(span["end"])))


def _copy_line(line):
    """Copy a subtitle line. Pipeline lines are ``SrtItem`` objects, and
    ``dict(item)`` fails because they iterate field names, not pairs.
    """
    if isinstance(line, dict):
        return dict(line)
    items = getattr(line, "items", None)
    if callable(items):
        return dict(items())
    return dict(line)


def _covered_spans(line, spans, min_overlap_ms):
    start = int(line["start_time"])
    end = int(line["end_time"])
    covered = []
    for span in spans:
        overlap = min(end, int(span["end"])) - max(start, int(span["start"]))
        if overlap < min_overlap_ms:
            continue
        piece = {
            "start": max(start, int(span["start"])),
            "end": min(end, int(span["end"])),
            "speaker": span["speaker"],
        }
        if covered and covered[-1]["speaker"] == piece["speaker"]:
            covered[-1]["end"] = max(covered[-1]["end"], piece["end"])
        else:
            covered.append(piece)
    return covered


def _cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _mean(left, right):
    return [(a + b) / 2 for a, b in zip(left, right)]


def _ms(value):
    value = max(int(value), 0)
    hours, rest = divmod(value, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
