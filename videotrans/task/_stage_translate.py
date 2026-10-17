import copy
import json
import shutil
import time
from pathlib import Path

from videotrans.configure.config import tr, logger
from videotrans.translator import run as run_trans
from videotrans.util.help_misc import vail_file
from videotrans.util.help_srt import get_subtitle_from_srt, delete_punc


class TranslateMixin:

    def trans(self) -> None:
        _st=time.time()
        if self._exit() or not self.should_trans: return

        self.precent += 3
        self.signal(text=tr('starttrans'))

        if vail_file(self.cfg.target_sub):
            self.signal(
                text=Path(self.cfg.target_sub).read_text(encoding="utf-8", errors="ignore"),
                type='replace_subtitle'
            )
            return

        rawsrt = get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
        self.signal(text=tr('kaishitiquhefanyi'))

        target_srt = run_trans(
            translate_type=self.cfg.translate_type,
            text_list=copy.deepcopy(rawsrt),
            uuid=self.uuid,
            source_code=self.cfg.source_language_code,
            target_code=self.cfg.target_language_code,
            is_cuda=self.cfg.is_cuda,
            skip_cache=bool(
                self.cfg.clear_cache or getattr(self, "force_new_dubbing", False)
            ),
            reference_pairs=getattr(self, "reference_pairs", None),
        )
        if self._exit():  return

        target_srt = self.check_target_sub(rawsrt, target_srt)
        from videotrans.task._song_cues import without_song_cues
        rawsrt, target_srt, dropped_at = without_song_cues(rawsrt, target_srt)
        if dropped_at:
            logger.debug(f"Springer {len(dropped_at)} sungne linjer over")
            self._save_srt_target(rawsrt, self.cfg.source_sub)
            self.source_srt_list = rawsrt
            self._drop_speakers_at(dropped_at, len(rawsrt) + len(dropped_at))
        # 删除原字幕标点
        if self.cfg.fix_punc==2:
            for it in rawsrt:
                it['text']=delete_punc(it['text'])
            self._save_srt_target(rawsrt, self.cfg.source_sub)
        # 无需配音或提取模式，在此移除标点
        if self.cfg.app_mode=='tiqu' or not self.should_dubbing:
            for it in target_srt:
                it['text']=it['text'].strip('...').strip('…').strip()
                if self.cfg.fix_punc==2:
                    it['text']=delete_punc(it['text'])


        if self.cfg.app_mode=='tiqu':
            shutil.copy2(self.cfg.source_sub, f"{self.cfg.target_dir}/{self.cfg.noextname}-{self.cfg.source_language_code}.srt")
            if self.cfg.output_srt > 0 and self.cfg.source_language_code != self.cfg.target_language_code:
                _source_srt_len = len(rawsrt)
                for i, it in enumerate(target_srt):
                    if i < _source_srt_len and self.cfg.output_srt == 1:
                        it['text'] = ("\n".join([rawsrt[i]['text'].strip(), it['text'].strip()])).strip()
                    elif i < _source_srt_len and self.cfg.output_srt == 2:
                        it['text'] = ("\n".join([it['text'].strip(), rawsrt[i]['text'].strip()])).strip()

        self._save_srt_target(target_srt, self.cfg.target_sub)

        if self.cfg.app_mode == 'tiqu':
            _output_file = f"{self.cfg.target_dir}/{self.cfg.noextname}.srt"
            if self.cfg.copysrt_rawvideo:
                p = Path(self.cfg.name)
                _output_file = f'{p.parent.as_posix()}/{p.stem}.srt'
            if not Path(_output_file).exists():
                shutil.copy2(self.cfg.target_sub, _output_file)

        self.signal(text=tr('endtrans'))
        logger.debug(f'[字幕翻译阶段结束耗时]:{time.time()-_st}s')

    def _drop_speakers_at(self, indexes, original_len):
        drop = set(indexes)
        for folder in (self.cfg.cache_folder, self.cfg.target_dir):
            path = Path(folder) / "speaker.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, list) or len(data) != original_len:
                continue
            path.write_text(
                json.dumps([item for i, item in enumerate(data) if i not in drop]),
                encoding="utf-8",
            )
