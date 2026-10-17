import copy
import hashlib
import re
import shutil
import time
from pathlib import Path

from videotrans.configure._paths import DUBBING_CACHE
from videotrans.configure.config import tr, app_cfg, settings, logger
from videotrans.configure.excepts import DubbingSrtError
from videotrans.tts import run as run_tts, SUPPORT_CLONE, CHATTERBOX_TTS
from videotrans.task._speaker_refs import source_index_for_cue
from videotrans.tts._designations import speakable_text
from videotrans.util.help_misc import get_md5, vail_file
from videotrans.util.help_srt import get_subtitle_from_srt, delete_punc


def dubbing_cache_key(*, text, voice, rate, volume, pitch, tts_type, silence, ref_wav=""):
    """Same words with another cloned voice must not reuse the old wav."""
    ref_id = ""
    path = Path(ref_wav) if ref_wav else None
    if path and path.is_file():
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        ref_id = digest.hexdigest()
    return get_md5(
        f"{text}-{voice}-{rate}-{volume}-{pitch}-{tts_type}-{silence}-{ref_id}"
    )


class DubbingMixin:

    def dubbing(self) -> None:
        _st=time.time()
        if self._exit() or self.cfg.app_mode == 'tiqu':
            return
        if self.should_dubbing:
            self.signal(text=tr('kaishipeiyin'))
        self.precent += 3
        self._tts()
        
        if  Path(self.cfg.source_sub).exists():
            logger.debug('配音结束后，移除原始字幕中所有标点')
            subs = get_subtitle_from_srt(self.cfg.source_sub)
            for it in subs:
                if self.cfg.fix_punc==2:
                    it['text']=delete_punc(it['text'])
                it['text']=it['text'].strip('...').strip('…').strip()
            self._save_srt_target(subs, self.cfg.source_sub)
        if self.should_dubbing:
            self.signal(text=tr('The dubbing is finished'))
            logger.debug(f'[语音合成阶段结束耗时]:{time.time()-_st}s')

    def _tts(self) -> None:
        if not self.should_dubbing:
            self.signal(text='Skip tts')
            return
        queue_tts = []
        subs = get_subtitle_from_srt(self.cfg.target_sub)
        source_subs = get_subtitle_from_srt(self.cfg.source_sub)
        if len(subs) < 1:
            raise DubbingSrtError(f"SRT file error:{self.cfg.target_sub}")
        try:
            rate = int(str(self.cfg.voice_rate).replace('%', ''))
        except (ValueError,TypeError):
            rate = 0

        rate = f"+{rate}%" if rate >= 0 else f"{rate}%"

        line_roles = app_cfg.line_roles
        voice_role = self.cfg.voice_role
        logger.debug(f'{line_roles=}')
        _lang=self.cfg.detect_language.split('-')[0]
        speakers = self._load_speaker_ids()
        for i, it in enumerate(subs):
            if it['end_time'] < it['start_time'] or not it['text'].strip():
                continue
            voice = line_roles.get(f'{it["line"]}', voice_role) if line_roles else voice_role
            src_index = source_index_for_cue(it, source_subs)
            src = source_subs[src_index] if src_index is not None else None
            speaker = speakers[src_index] if src_index is not None and src_index < len(speakers) else ""
            if src_index is not None and src_index != i:
                logger.debug(f"linje {it['line']} bruger taler {speaker} fra kildelinje {src_index + 1}")

            tmp_dict = {
                "text": it['text'],
                "line": it['line'],
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "startraw": it['startraw'],
                "endraw": it['endraw'],
                "ref_text": src['text'] if src else '',
                "start_time_source": src['start_time'] if src else it['start_time'],
                "end_time_source": src['end_time'] if src else it['end_time'],
                "speaker": speaker,
                "role": voice,
                "rate": rate,
                "volume": self.cfg.volume,
                "pitch": self.cfg.pitch,
                "tts_type": self.cfg.tts_type,
                "filename": f"{self.cfg.cache_folder}/{i}.wav"
            }
            if str(voice).strip().lower() == 'clone' and self.cfg.tts_type in SUPPORT_CLONE:
                tmp_dict['ref_wav'] = f"{self.cfg.cache_folder}/clone-{i}.wav"
                tmp_dict['ref_language'] = _lang
            if self.cfg.tts_type == CHATTERBOX_TTS:
                spoken = speakable_text(tmp_dict["text"])
                if spoken != tmp_dict["text"]:
                    tmp_dict["speak_text"] = spoken
                    logger.debug(f"Chatterbox staver linje {tmp_dict['line']}: {spoken}")
            queue_tts.append(tmp_dict)

        self.queue_tts = copy.deepcopy(queue_tts)

        if not self.queue_tts or len(self.queue_tts) < 1:
            raise RuntimeError(f'字幕长度为0，无法继续配音')

        if len([it.get("ref_wav") for it in self.queue_tts if it.get("ref_wav")]) > 0:
            self._create_ref_from_vocal()
        self._reuse_dubbing_cache()

        run_tts(
            queue_tts=self.queue_tts,
            language=self.cfg.target_language_code,
            uuid=self.uuid,
            tts_type=self.cfg.tts_type,
            is_cuda=self.cfg.is_cuda
        )
        outname=None
        if settings.get('save_segment_audio', False):
            outname = self.cfg.target_dir + f'/segment_audio_{self.cfg.noextname}'
            Path(outname).mkdir(parents=True, exist_ok=True)
        for it in self.queue_tts:
            it['text']=it['text'].strip('...').strip('…').strip()
            if self.cfg.fix_punc==2:
                it['text']=delete_punc(it['text'])
            if Path(it['filename']).exists():
                # 保存缓存
                shutil.copy2(it['filename'],f'{DUBBING_CACHE}/'+Path(it['filename']).name.split('-')[-1])
                if outname:
                    text = re.sub(r'["\'*?\\/|:<>\r\n\t]+', '', it['text'], flags=re.I | re.S)
                    name = f'{outname}/{it["line"]}-{text[:60]}.wav'
                    shutil.copy2(it['filename'], name)
        
        self._save_srt_target(self.queue_tts, self.cfg.target_sub)

    def _reuse_dubbing_cache(self) -> None:
        silence = (
            settings.get("remove_dubb_silence", True),
            settings.get("remove_dubb_all_silence", True),
            settings.get("remove_dubb_silence_level", "default"),
        )
        for index, it in enumerate(self.queue_tts):
            key = dubbing_cache_key(
                text=f"{self.cfg.target_language_code}-{it.get('speak_text') or it['text']}",
                voice=it.get("role"),
                rate=it.get("rate"),
                volume=it.get("volume"),
                pitch=it.get("pitch"),
                tts_type=it.get("tts_type"),
                silence=silence,
                ref_wav=it.get("ref_wav") or "",
            )
            it["filename"] = f"{self.cfg.cache_folder}/{index}-{key}.wav"
            cached = f"{DUBBING_CACHE}/{key}.wav"
            if (
                self.cfg.clear_cache
                or getattr(self, "force_new_dubbing", False)
                or not vail_file(cached)
            ):
                continue
            shutil.copy2(cached, it["filename"])
            logger.debug(f"复用配音缓存 行{it.get('line')}")