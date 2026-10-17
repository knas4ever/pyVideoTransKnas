"""Wipe one run's output and the caches that would replay old work."""

import shutil
from pathlib import Path

from videotrans.configure._paths import DUBBING_CACHE, TRANSLATE_CACHE
from videotrans.configure.config import logger


def empty_directory(folder):
    """Remove everything inside ``folder``. The folder itself stays."""
    path = Path(folder) if folder else None
    if path is None or not path.is_dir():
        return 0
    removed = 0
    for child in list(path.iterdir()):
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except OSError as exc:
            logger.error(f"clear failed {child}: {exc}")
    return removed


def clear_previous_run(target_dir, cache_folder):
    """Empty this video's output, its work folder, and the shared voice and translation caches.

    Model files and test clips live elsewhere and are left in place.
    """
    removed = 0
    for folder in (target_dir, cache_folder, DUBBING_CACHE, TRANSLATE_CACHE):
        removed += empty_directory(folder)
    Path(DUBBING_CACHE).mkdir(parents=True, exist_ok=True)
    Path(TRANSLATE_CACHE).mkdir(parents=True, exist_ok=True)
    logger.debug(
        f"已清理输出和缓存 target={target_dir} cache={cache_folder} removed={removed}"
    )
    return removed
