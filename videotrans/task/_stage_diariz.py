import time
from pathlib import Path

from videotrans.configure.config import logger


def reuses_saved_speakers(*, clear_cache, speaker_json_exists):
    """A cleared run assigns speakers again. An existing file is only kept otherwise."""
    if clear_cache:
        return False
    return bool(speaker_json_exists)


class DiarizMixin:

    def diariz(self):
        _st=time.time()
        speaker_cache = Path(self.cfg.cache_folder) / "speaker.json"
        speaker_out = Path(self.cfg.target_dir) / "speaker.json"
        if self.cfg.clear_cache:
            speaker_cache.unlink(missing_ok=True)
            speaker_out.unlink(missing_ok=True)
            logger.debug("已选清理，忽略已有说话人，重新分离")
        # 只要 do_diarize 是 False,无论是否选中都不分离说话人
        if self._exit() or not self.should_dubbing or not self.do_diarize or not self.cfg.enable_diariz or self.max_speakers == 1 or reuses_saved_speakers(
                clear_cache=self.cfg.clear_cache,
                speaker_json_exists=speaker_cache.exists(),
        ):
            return

        self._diariz_common()