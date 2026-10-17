"""Pick one reference span per speaker for local voice cloning."""
import re
import shutil
from pathlib import Path


def source_index_for_cue(cue, sources):
    """Index of the source cue that shares the most time with ``cue``.

    A translation can merge two spoken lines into one. The target list is then
    shorter, and pairing by position gives the following line the previous voice.
    """
    if not sources:
        return None
    start = int(cue.get("start_time") or 0)
    end = int(cue.get("end_time") or 0)
    best = None
    best_overlap = 0
    for index, src in enumerate(sources):
        overlap = min(end, int(src.get("end_time") or 0)) - max(start, int(src.get("start_time") or 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best = index
    if best is not None:
        return best
    return min(
        range(len(sources)),
        key=lambda index: abs(int(sources[index].get("start_time") or 0) - start),
    )


# A pause longer than this starts a new scene for the same speaker label.
SCENE_GAP_MS = 8000
# A line this long, from another scene, keeps its own voice.
OWN_VOICE_MS = 1500


def plan_speaker_reference_spans(spans, speakers, *, max_ms=12000, min_piece_ms=400):
    """Return ``{speaker: [(start_ms, end_ms), ...]}``.

    Spans and speakers are parallel. A plan is returned only when at least two
    speakers are present. The reference comes from the scene with the most
    lines, in the order they were spoken. One later long line must not lead
    the clip: the clone hears only the first seconds. An empty dict means the
    caller should keep one clip per line.
    """
    if not speakers or not spans:
        return {}
    grouped = {}
    for i, span in enumerate(spans):
        if i >= len(speakers) or not speakers[i]:
            continue
        start, end = int(span[0]), int(span[1])
        if end <= start:
            continue
        grouped.setdefault(str(speakers[i]), []).append((start, end))
    if len(grouped) < 2:
        return {}

    plan = {}
    for spk, pieces in grouped.items():
        scene = _main_scene(pieces)
        chosen = []
        total = 0
        for start, end in scene:
            room = max_ms - total
            if room < min_piece_ms:
                break
            duration = end - start
            if duration < min_piece_ms:
                continue
            chosen.append((start, start + min(duration, room)))
            total += chosen[-1][1] - chosen[-1][0]
        if not chosen and scene:
            start, end = scene[0]
            if end > start:
                chosen = [(start, min(end, start + max_ms))]
        if chosen:
            plan[spk] = chosen
    return plan


def clone_span_ms(item, *, per_sentence):
    """Source cue times when each sentence keeps its own clone prompt.

    ``None`` means the caller should cut the target subtitle window.
    """
    if not per_sentence:
        return None
    try:
        start = int(item.get("start_time_source"))
        end = int(item.get("end_time_source"))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def line_uses_own_voice(start, end, reference_pieces, *, own_ms=OWN_VOICE_MS):
    """A long line from another scene is not spoken with the shared reference."""
    start, end = int(start), int(end)
    if end - start < own_ms:
        return False
    for ref_start, ref_end in reference_pieces or []:
        if min(end, int(ref_end)) - max(start, int(ref_start)) > 0:
            return False
    return True


def _main_scene(pieces):
    """The run of lines with the most cues. Ties keep the one with more speech."""
    ordered = sorted(pieces, key=lambda item: item[0])
    scenes = []
    current = []
    for start, end in ordered:
        if current and start - current[-1][1] > SCENE_GAP_MS:
            scenes.append(current)
            current = []
        current.append((start, end))
    if current:
        scenes.append(current)
    return max(scenes, key=lambda scene: (len(scene), sum(end - start for start, end in scene)))


def speaker_cap_to_nums(cap):
    """Turn a speaker cap from the web form into ``nums_diariz``.

    The task adds one to any positive value, and the desktop list stores that
    index. 0 stays unlimited. 8 speakers is stored as 7.
    """
    try:
        cap = int(cap or 0)
    except (TypeError, ValueError):
        return 0
    if cap <= 0:
        return 0
    return max(cap, 2) - 1


def reference_cut_workers(job_count, cpu_count, requested=0):
    """How many speaker clips to cut at once.

    ``requested`` 0 follows the CPU and stays at 8 or below, so a 12-core
    machine uses 8. A positive number overrides that, but not past the CPU
    count or the number of speakers.
    """
    jobs = max(int(job_count or 0), 0)
    if jobs < 1:
        return 1
    cpus = max(int(cpu_count or 1), 1)
    try:
        asked = int(requested or 0)
    except (TypeError, ValueError):
        asked = 0
    if asked > 0:
        return max(1, min(asked, jobs, cpus))
    auto = cpus - 1 if cpus > 1 else 1
    return max(1, min(8, auto, jobs))


def write_speaker_reference(*, speaker, pieces, cache_folder, vocal, cut, concat, to_time):
    """Cut one speaker's pieces. Return the reference path, or None."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(speaker)) or "spk"
    ref_path = Path(cache_folder) / f"spkref-{safe}.wav"
    part_paths = []
    for index, (start, end) in enumerate(pieces):
        part = Path(cache_folder) / f"spkref-{safe}-{index}.wav"
        if cut(
            audio_file=vocal,
            ss=to_time(start),
            to=to_time(end),
            out_file=str(part),
        ):
            part_paths.append(part)
    if not part_paths:
        return None
    if ref_path.exists():
        ref_path.unlink()
    if len(part_paths) == 1:
        shutil.move(part_paths[0], ref_path)
    else:
        list_file = Path(cache_folder) / f"spkref-{safe}.txt"
        list_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in part_paths),
            encoding="utf-8",
        )
        if not concat(str(list_file), str(ref_path)):
            return None
    return str(ref_path) if ref_path.is_file() else None
