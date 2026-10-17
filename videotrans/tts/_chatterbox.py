import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

from videotrans.configure._i18n import tr
from videotrans.configure._paths import REDUBB_STATUS_FILE, REDUBB_QUEUE_FILE
from videotrans.configure.config import params, logger, app_cfg, ROOT_DIR, settings
from videotrans.configure.excepts import DubbingSrtError
from videotrans.configure.constants import SPACY_URL_MS,SPACY_URL_HF
from videotrans.tts._base import BaseTTS
from videotrans.util.help_role import get_chatterbox_role
from videotrans.util.gpus import mps_or_cpu
import soundfile as sf
from videotrans.util.help_misc import vail_file, is_connect_hf


@dataclass(repr=False)
class ChatterBoxTTS(BaseTTS):
    def __post_init__(self):
        super().__post_init__()
        self.roledict = get_chatterbox_role()
        self.local_dir=f'{ROOT_DIR}/models/models--resembleAI--chatterbox'

    def _download(self):
        if Path(f'{self.local_dir}/ve.pt').exists(): return True
        from videotrans.util.help_down import check_and_down_hf,check_and_down_ms,down_zip
        down_zip(f'{ROOT_DIR}/models/pkuser_home/spacy_ontonotes',SPACY_URL_HF if is_connect_hf() else SPACY_URL_MS)
        
        check_and_down_hf("", 'resembleAI/chatterbox', self.local_dir,
                                callback=self._process_callback,
                                allow_list=["ve.pt","t3_mtl23ls_v3.safetensors","t3_mtl23ls_v2.safetensors","s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"])
        return True

    def _exec(self):
        # Re-dubbing keeps the in-process loop. A film is dubbed in a fresh
        # process per batch so the model is not added to the WebUI's memory.
        if not self.is_redubb:
            return self._exec_batched()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        import inspect
        device=settings.get('device_name','auto')
        if device=='auto':
            device='cuda' if self.is_cuda else mps_or_cpu()
        logger.debug(f'Chatterbox running on {device}')
        load_kwargs = {"device": device}
        if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_local).parameters:
            load_kwargs["t3_model"] = "v3"
        model = ChatterboxMultilingualTTS.from_local(self.local_dir, **load_kwargs)

        ok, err = 0, 0
        _except = None
        cfg_weight = float(params.get("chatterbox_cfg_weight", '0.5'))
        exaggeration = float(params.get("chatterbox_exaggeration", '0.5'))
        lang = self.language.split('-')[0]
        queue_tts=self.queue_tts
        while 1:
            if self.is_redubb and Path(REDUBB_STATUS_FILE).exists():
                return True
            if self.is_redubb:
                try:
                    queue_tts=json.loads(Path(REDUBB_QUEUE_FILE).read_text(encoding='utf-8'))
                except (OSError,json.JSONDecodeError) as e:
                    logger.exception(f'supertonic-3: {e}',exc_info=True)
                    raise


            for i,item in enumerate(queue_tts):
                if app_cfg.exit_soft or (self.is_redubb and Path(REDUBB_STATUS_FILE).exists()):
                    return True
                if vail_file(item['filename']):
                    continue
                ref_wav = None
                try:
                    ref_wav, _ = self.get_ref_wav(item)
                except Exception:
                    logger.warn('无参考音频，使用内置音色')
                try:
                    self.signal(text=f'{tr("Dubbing")} {i}/{self.len}')
                    wav_tensor = model.generate(item.get('speak_text') or item['text'], exaggeration=exaggeration, cfg_weight=cfg_weight,
                                                language_id=lang, audio_prompt_path=ref_wav)
                    wav_tensor = wav_tensor.detach().cpu()
                    if wav_tensor.ndim == 2:
                        wav_np = wav_tensor.transpose(0, 1).numpy()
                    else:
                        wav_np = wav_tensor.numpy()
                    # 写入 WAV 格式到内存
                    sf.write(item['filename'] + "-24k.wav", wav_np, model.sr, format='wav')
                    if not vail_file(item['filename'] + '-24k.wav'):
                        err += 1
                        continue
                    ok += 1
                    self.convert_to_wav(item['filename'] + '-24k.wav', item['filename'])
                except Exception as e:
                    _except = e
                    logger.exception(f'chatterbox dubbing error:{e}', exc_info=True)
                    err += 1
            if self.is_redubb:
                time.sleep(0.5)
                continue
            break

        if ok == 0:
            raise _except if _except else DubbingSrtError('ChatterBox-TTS dubbing error')

        msg = f"{tr('Dubbing')} ended"
        if err > 0 and ok > 0:
            msg = f'[{err}] errors, {ok} succeed'

        self.signal(text=msg)
        logger.debug(f'ChatterBox 配音结束：{msg}')

    def _exec_batched(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        pending = [it for it in self.queue_tts if not vail_file(it.get("filename"))]
        batch = 20
        cfg_weight = float(params.get("chatterbox_cfg_weight", "0.5"))
        exaggeration = float(params.get("chatterbox_exaggeration", "0.5"))
        total = len(self.queue_tts)
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            title = f"ChatterBox {start + 1}-{start + len(chunk)}/{total}"
            self._run_chunk(chunk, title, cfg_weight, exaggeration)
            gc.collect()

    def _run_chunk(self, chunk, title, cfg_weight, exaggeration):
        from videotrans.process.chatterbox_tts import _write_silence, chatterbox_fun

        queue_file = f"{Path(chunk[0]['filename']).parent}/chatterbox-{time.time()}.json"
        Path(queue_file).write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        try:
            self._new_process(
                callback=chatterbox_fun,
                title=title,
                is_cuda=self.is_cuda,
                kwargs={
                    "queue_tts_file": queue_file,
                    "language": self.language.split("-")[0],
                    "is_cuda": self.is_cuda,
                    "local_dir": self.local_dir,
                    "cfg_weight": cfg_weight,
                    "exaggeration": exaggeration,
                },
            )
        except Exception as exc:
            logger.exception(f"{title} failed: {exc}")
            if len(chunk) == 1:
                _write_silence(chunk[0])
                logger.error(
                    f"Chatterbox line left silent: {(chunk[0].get('text') or '')[:40]}"
                )
                return
            logger.error(f"{title} crashed. Retrying {len(chunk)} lines one at a time.")
            for index, item in enumerate(chunk, start=1):
                if vail_file(item.get("filename")):
                    continue
                self._run_chunk([item], f"{title} line {index}", cfg_weight, exaggeration)
