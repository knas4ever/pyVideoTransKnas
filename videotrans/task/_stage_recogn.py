import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from videotrans.configure.config import tr, ROOT_DIR, settings, logger, params
from videotrans.configure.constants import DENOISE_URL_MS, PUNC_RESTORE_MS, DENOISE_URL_HF, PUNC_RESTORE_HF
from videotrans.configure.excepts import SpeechToTextError
from videotrans.recognition import run as run_recogn,  FASTER_WHISPER
from videotrans.util.help_ffmpeg import conver_to_16k, runffmpeg, cut_from_audio
from videotrans.util.help_misc import vail_file, is_connect_hf
from videotrans.util.help_srt import get_subtitle_from_srt, delete_punc


class RecognMixin:

    def recogn(self) -> None:
        _st=time.time()
        if self._exit(): return
        if not self.should_recogn: return
        self.precent += 3
        self.signal(text=tr("kaishishibie"))
        if vail_file(self.cfg.source_sub):
            logger.debug(f'复用已存在的原始字幕，跳过识别: {self.cfg.source_sub}')
            self.source_srt_list = get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            saved_speakers = Path(self.cfg.target_dir + "/speaker.json")
            if saved_speakers.exists() and not self.cfg.clear_cache:
                shutil.copy2(saved_speakers, self.cfg.cache_folder + "/speaker.json")
            self._recogn_succeed()
            return

        if not vail_file(self.cfg.source_wav):
            raise SpeechToTextError(tr("Failed to separate audio, please check the log or retry"))
        # 未分离人声背景时才降噪，已分离则不再降噪
        if self.cfg.remove_noise and not self.cfg.is_separate:
            from videotrans.util.help_down import down_file_from_hf
            _remove_noise_wav = f"{self.cfg.cache_folder}/remove_noise.wav"
            if vail_file(_remove_noise_wav):
                self.cfg.source_wav = _remove_noise_wav
                self.clone_ref = _remove_noise_wav
                logger.debug(f'复用已存在的降噪缓存文件')
            else:
                title = tr("Starting to process speech noise reduction, which may take a long time, please be patient")
                kw = {
                    "input_file": self.cfg.source_wav if not self.cfg.vocal or not Path(self.cfg.vocal).exists() else self.cfg.vocal,
                    "output_file": _remove_noise_wav,
                    "is_cuda": self.cfg.is_cuda
                }
                try:
                    down_file_from_hf(f'{ROOT_DIR}/models/onnx', urls=DENOISE_URL_MS if not is_connect_hf() else DENOISE_URL_HF,
                                            callback=self._process_callback)
                    from videotrans.process.prepare_audio import remove_noise
                    _rs = self._new_process(callback=remove_noise, title=title, is_cuda=self.cfg.is_cuda, kwargs=kw)
                    if _rs:
                        self.clone_ref = _remove_noise_wav
                        self.cfg.source_wav = _remove_noise_wav
                    self.signal(text='remove noise end')
                except Exception as e:
                    logger.exception(f'降噪失败，跳过 {e}', exc_info=True)

        self.signal(text=tr("Speech Recognition to Word Processing"))
        raw_subtitles = run_recogn(
            recogn_type=self.cfg.recogn_type,
            uuid=self.uuid,
            model_name=self.cfg.model_name,
            audio_file=self.cfg.source_wav,
            detect_language=self.cfg.detect_language,
            cache_folder=self.cfg.cache_folder,
            is_cuda=self.cfg.is_cuda,
            subtitle_type=self.cfg.subtitle_type,
            max_speakers=self.max_speakers,
        )
        if self._exit(): return
        if not raw_subtitles:
            raise SpeechToTextError(self.cfg.basename + tr('recogn result is empty'))

        if self.cfg.app_mode=='tiqu' and not self.should_trans and self.cfg.fix_punc==2:
            logger.debug('仅提取不翻译模式下，移除所有标点')
            for it in raw_subtitles:
                it['text'] = delete_punc(it['text'])

        self._save_srt_target(raw_subtitles, self.cfg.source_sub)
        self.source_srt_list = raw_subtitles
        # 恢复标点
        if self.cfg.fix_punc==1:
            try:
                down_file_from_hf(f'{ROOT_DIR}/models/puntc', PUNC_RESTORE_MS if not is_connect_hf() else PUNC_RESTORE_HF, callback=self._process_callback)
                from videotrans.process.prepare_audio import fix_punc
                text_dict = {f'{it["line"]}': re.sub(r'[,.?!，。？！]', ' ', it["text"]) for it in self.source_srt_list}
                text_dict_file=f'{self.cfg.cache_folder}/text_dict_file_{time.time()}.json'
                Path(text_dict_file).write_text(json.dumps(text_dict),encoding="utf-8")
                kw = {"text_dict_file": text_dict_file, "is_cuda": self.cfg.is_cuda}
                _rs = self._new_process(callback=fix_punc, title=tr("Restoring punct"), is_cuda=self.cfg.is_cuda,
                                        kwargs=kw)
                if _rs:
                    text_dict_obj=json.loads(Path(text_dict_file).read_text(encoding='utf-8'))
                    for it in self.source_srt_list:
                        it['text'] = text_dict_obj.get(f'{it["line"]}', it['text'])
                        if self.cfg.detect_language.split('-')[0] == 'en':
                            it['text'] = it['text'].replace('，', ',').replace('。', '. ').replace('？', '?').replace('！','!')
                    self._save_srt_target(self.source_srt_list, self.cfg.source_sub)
                else:
                    logger.error('标点恢复出错了，跳过')
            except Exception as e:
                logger.exception(f'标点恢复失败，跳过 {e}', exc_info=True)

        self.signal(text=Path(self.cfg.source_sub).read_text(encoding='utf-8'), type='replace_subtitle')


        # LLM纠错
        if self.cfg.rephrase:
            self.source_srt_list=self._llmpost(self.source_srt_list)
        
        self._recogn_succeed()
        self.signal(text=tr('endtiquzimu'))
        logger.debug(f'[语音识别阶段结束耗时]:{time.time()-_st}s')

    def _recogn_succeed(self) -> None:
        self.precent += 5
        if self.cfg.app_mode == 'tiqu' and not self.should_trans:
            shutil.copy2(self.cfg.source_sub,  f"{self.cfg.target_dir}/{self.cfg.noextname}.srt")
        self.signal(text=tr('endtiquzimu'))

    # 二次识别固定使用 faster-whisper 渠道
    def recogn2pass(self) -> None:
        _st=time.time()
        if not self.should_recogn2 or self._exit():
            return
        if not vail_file(self.cfg.target_wav):
            logger.debug(f'跳过二次识别，因无配音音频文件')
            return

        self.precent += 3
        self.signal(text=tr("Secondary speech recognition of dubbing files"))

        shibie_audio = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.wav'
        outsrt_file = f'{self.cfg.cache_folder}/recogn2pass-{time.time()}.srt'
        try:
            conver_to_16k(self.cfg.target_wav, shibie_audio)
        except Exception as e:
            logger.exception(f'二次识别配音音频生成字幕时，预处理音频失败，静默跳过 {e}', exc_info=True)
            return

        if not vail_file(shibie_audio):
            logger.error(f'二次识别配音音频生成字幕时，预处理音频失败，静默跳过')
            return

        try:

            detect_language = self.cfg.target_language_code.split('-')[0]
            recogn_type = FASTER_WHISPER
            model_name = settings.get('model_for_recogn2','large-v3-turbo')
            logger.debug(f'二次识别：faster-whisper + {detect_language=} + {model_name=}')
            raw_subtitles = run_recogn(
                recogn_type=recogn_type,
                uuid=self.uuid,
                model_name=model_name,
                audio_file=shibie_audio,
                detect_language=detect_language,
                cache_folder=self.cfg.cache_folder,
                is_cuda=self.cfg.is_cuda,
                recogn2pass=True
            )
            if self._exit(): return
            if not raw_subtitles:
                logger.error('二次识别出错：' + tr('recogn result is empty'))
                return

            if self.cfg.rephrase:
                raw_subtitles=self._llmpost(raw_subtitles,'2')

            if self.cfg.fix_punc==2:
                logger.debug('二次识别后，移除所有标点')
                for it in raw_subtitles:
                    it['text']=delete_punc(it['text'])
            self._save_srt_target(raw_subtitles, outsrt_file)

            if not vail_file(outsrt_file):
                logger.error(f'二次识别配音文件失败，原因未知')
                return
            shutil.copy2(outsrt_file, self.cfg.target_sub)
            self.signal(text='STT 2 pass end')
            logger.debug('二次识别成功完成')
        except Exception as e:
            logger.exception(f'二次识别配音音频生成字幕时失败，静默跳过 {e}', exc_info=True)
            return
        logger.debug(f'[二次识别阶段结束耗时]:{time.time()-_st}s')



    def _create_ref_from_vocal(self):
        vocal = self.cfg.source_wav
        if self.clone_ref and Path(self.clone_ref).exists():
            vocal=self.clone_ref
        else:
            try:
                tmpfile = self.cfg.cache_folder + "/clone_ref_44100.wav"
                runffmpeg([
                    "-y",
                    "-i",
                    self.cfg.name,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-c:a",
                    "pcm_s16le",
                    tmpfile
                ])
                vocal=tmpfile
            except Exception as e:
                logger.exception(f'克隆语音前分离出 44.1k 的原始音频失败',exc_info=True)

        logger.debug(f'语音克隆模式下，所用参考音频为:{vocal}')
        per_sentence = bool(getattr(self.cfg, "voice_per_sentence", False))
        if per_sentence:
            logger.debug("Hver sætning klones fra sin egen lyd")
            shared_refs = set()
        else:
            shared_refs = self._assign_speaker_reference_wavs(vocal)
        from videotrans.task._speaker_refs import clone_span_ms
        from videotrans.util.help_srt import ms_to_time_string

        def _cutaudio_from_vocal(it):
            try:
                span = clone_span_ms(it, per_sentence=per_sentence)
                if span is None:
                    ss, to = it["startraw"], it["endraw"]
                else:
                    ss = ms_to_time_string(ms=span[0])
                    to = ms_to_time_string(ms=span[1])
                logger.debug(f"裁切对应片段为参考音频：{ss}->{to}\n当前{it=}")
                cut_from_audio(
                    audio_file=vocal,
                    ss=ss,
                    to=to,
                    out_file=it['ref_wav']
                )
            except Exception as e:
                logger.exception(f'裁切参考音频失败:{it=},{e}', exc_info=True)

        all_task = []
        with ThreadPoolExecutor(max_workers=min(8, len(self.queue_tts), os.cpu_count())) as pool:
            for item in self.queue_tts:
                if item.get('ref_wav') and item['ref_wav'] not in shared_refs:
                    all_task.append(pool.submit(_cutaudio_from_vocal, item))
            if len(all_task) > 0:
                _ = [i.result() for i in all_task]

    def _load_speaker_ids(self):
        for folder in (self.cfg.cache_folder, self.cfg.target_dir):
            path = Path(folder) / "speaker.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list) and data:
                return [str(item) if item else "" for item in data]
        return []

    def _assign_speaker_reference_wavs(self, vocal):
        """One reference clip per detected speaker. Returns the written paths."""
        loaded = self._load_speaker_ids()
        speakers = []
        for index, it in enumerate(self.queue_tts):
            # The dubbing queue stores the speaker matched by time. The file is
            # still ordered like the source cues, so an index into it shifts
            # after a merged translation line.
            speaker = it.get("speaker")
            if speaker:
                speakers.append(str(speaker))
            elif index < len(loaded):
                speakers.append(loaded[index])
            else:
                speakers.append("")
        if len(set(speakers)) < 2 or not vocal or not Path(vocal).exists():
            return set()
        from videotrans.task._speaker_refs import (
            line_uses_own_voice,
            plan_speaker_reference_spans,
        )
        from videotrans.util.help_srt import ms_to_time_string

        spans = []
        for it in self.queue_tts:
            if not it.get("ref_wav"):
                spans.append((0, 0))
                continue
            start = int(it.get("start_time_source") or it.get("start_time") or 0)
            end = int(it.get("end_time_source") or it.get("end_time") or 0)
            spans.append((start, end))
        plan = plan_speaker_reference_spans(spans, speakers)
        if not plan:
            return set()
        from videotrans.task._speaker_refs import reference_cut_workers, write_speaker_reference

        workers = reference_cut_workers(
            len(plan), os.cpu_count(), params.get("speaker_ref_threads", 0)
        )
        logger.debug(f'并行裁切说话人参考音频: {len(plan)} 人, {workers} 线程')

        def _concat(list_file, ref_path):
            return runffmpeg(
                ["-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", ref_path]
            )

        def _one(item):
            spk, pieces = item
            try:
                ref_path = write_speaker_reference(
                    speaker=spk,
                    pieces=pieces,
                    cache_folder=self.cfg.cache_folder,
                    vocal=vocal,
                    cut=cut_from_audio,
                    concat=_concat,
                    to_time=lambda ms: ms_to_time_string(ms=ms),
                )
            except Exception as e:
                logger.exception(f'按说话人裁切参考音频失败:{spk},{e}', exc_info=True)
                return spk, None, len(pieces)
            return spk, ref_path, len(pieces)

        written = set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, list(plan.items())))
        for spk, ref_path, piece_count in results:
            if not ref_path or not Path(ref_path).exists():
                continue
            for i, it in enumerate(self.queue_tts):
                if not it.get("ref_wav") or i >= len(speakers) or speakers[i] != spk:
                    continue
                start = int(it.get("start_time_source") or it.get("start_time") or 0)
                end = int(it.get("end_time_source") or it.get("end_time") or 0)
                if line_uses_own_voice(start, end, plan.get(spk)):
                    continue
                it["ref_wav"] = ref_path
            written.add(ref_path)
            logger.debug(f'说话人 {spk} 使用共用参考音频 {ref_path}，片段数 {piece_count}')
        return written
