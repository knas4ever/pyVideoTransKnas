"""Place each dubbed line in the middle of its original cue."""

import math

# Past this speed a line sounds clipped. Unused time in the previous cue can
# be taken so the line starts earlier instead of being sped that hard.
COMFORT_SPEED = 1.3
MAX_EARLY_MS = 2000


def centered_start_ms(start_time, end_time, dub_ms):
    """Start time that puts the dub in the middle of the original cue.

    A shorter line starts later. A longer line starts earlier by half of the
    extra length, so it also ends only half of that extra length late.
    """
    start = int(start_time)
    slot = max(0, int(end_time) - start)
    dub = max(0, int(dub_ms))
    return max(0, start + (slot - dub) // 2)


def early_start_slots(slots, dubs, comfort=COMFORT_SPEED, max_early_ms=MAX_EARLY_MS):
    """Lengthen a cramped slot by starting it earlier.

    The extra time comes only from the previous slot's unused tail, and never
    more than ``max_early_ms``. The total length stays the same. A line whose
    previous cue is already full of speech is left as it is.
    """
    widths = [max(0, int(slot)) for slot in slots]
    lengths = [max(0, int(dub)) for dub in dubs]
    borrows = []
    for index in range(1, len(widths)):
        room = widths[index]
        if room <= 0 or lengths[index] <= room * comfort:
            continue
        wanted = math.ceil(lengths[index] / comfort) - room
        spare = widths[index - 1] - lengths[index - 1]
        borrow = min(max(0, spare), max(0, wanted), max_early_ms)
        if borrow <= 0:
            continue
        widths[index - 1] -= borrow
        widths[index] += borrow
        borrows.append((index, borrow))
    return widths, borrows


def move_long_lines_earlier(items, comfort=COMFORT_SPEED, max_early_ms=MAX_EARLY_MS):
    """Apply :func:`early_start_slots` to abutted cue dicts."""
    if len(items) < 2:
        return []
    slots = [int(it["end_time"]) - int(it["start_time"]) for it in items]
    dubs = [int(it.get("dubb_time") or 0) for it in items]
    widths, borrows = early_start_slots(slots, dubs, comfort=comfort, max_early_ms=max_early_ms)
    if not borrows:
        return []
    cursor = int(items[0]["start_time"])
    for it, width in zip(items, widths):
        it["start_time"] = cursor
        it["end_time"] = cursor + width
        it["start_time_source"] = cursor
        it["end_time_source"] = cursor + width
        it["source_duration"] = width
        cursor += width
    return borrows


def silence_before_centered(timeline_ms, start_time, end_time, dub_ms):
    """Silence to insert so this line can reach its centered start.

    When the previous line still runs past that point, insert nothing. The
    whole previous sentence stays, and this one starts as soon as it can.
    """
    desired = centered_start_ms(start_time, end_time, dub_ms)
    return max(0, desired - int(timeline_ms))
