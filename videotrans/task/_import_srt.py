"""Place an uploaded subtitle file where recognition or translation will reuse it."""

from pathlib import Path

from videotrans.task.taskcfg import SrtItem
from videotrans.util._srt_parse import get_srt_from_list, get_subtitle_from_srt, ms_to_time_string


def install_subtitle_files(
    spoken_path,
    target_path,
    source_sub,
    target_sub,
    *,
    limit_ms=None,
    drop_paths=(),
):
    """Copy uploaded srt files into the job.

    A spoken file replaces the source subtitles. A target file replaces the
    translation. ``limit_ms`` keeps only cues that start inside a test clip.
    ``drop_paths`` are old speaker and word files that must not be reused.
    """
    spoken = _install_one(spoken_path, source_sub, limit_ms)
    target = _install_one(target_path, target_sub, limit_ms)
    if spoken is not None:
        for path in drop_paths:
            Path(path).unlink(missing_ok=True)
        if target is None:
            Path(target_sub).unlink(missing_ok=True)
    return {"spoken": spoken, "target": target}


def _install_one(upload, destination, limit_ms):
    path = _upload_path(upload)
    if path is None:
        return None
    lines = get_subtitle_from_srt(path.as_posix(), is_file=True)
    if limit_ms:
        lines = _within_limit(lines, int(limit_ms))
    if not lines:
        raise ValueError(path.name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(get_srt_from_list(lines), encoding="utf-8")
    return len(lines)


def _upload_path(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _within_limit(lines, limit_ms):
    kept = []
    for line in lines:
        start = int(line["start_time"])
        end = int(line["end_time"])
        if start >= limit_ms:
            continue
        if end > limit_ms:
            copied = SrtItem(**dict(line.items()))
            copied["end_time"] = limit_ms
            copied["endraw"] = ms_to_time_string(ms=limit_ms)
            copied["time"] = f"{copied['startraw']} --> {copied['endraw']}"
            line = copied
        kept.append(line)
    return kept
