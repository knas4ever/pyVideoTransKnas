# -*- coding: utf-8 -*-
import traceback, time
from videotrans.configure.config import ROOT_DIR, logger, settings
from pathlib import Path
from videotrans.process._audio_utils import _write_log

# A full film as one tensor crashes sherpa. One minute plus a short overlap
# stays small in memory and hides the cut.
_CHUNK_SECONDS = 60
_OVERLAP_SECONDS = 1.0


def vocal_bgm(*, input_file, vocal_file, instr_file, logs_file=None, uvr_models="UVR-MDX-NET-Inst_HQ_4",**kw):
    if uvr_models.startswith('spleeter'):
        return vocal_bgm_spleeter(input_file=input_file, vocal_file=vocal_file, instr_file=instr_file,
                                  logs_file=logs_file)

    import sherpa_onnx

    def create_offline_source_separation():
        model = f"{ROOT_DIR}/models/onnx/{uvr_models}.onnx"

        if not Path(model).is_file():
            raise ValueError(f"{model} does not exist.")

        _cf = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(
                    model=model,
                ),
                num_threads=int(settings.get('noise_separate_nums', 4)),
                debug=False,
                provider="cpu",
            )
        )
        if not _cf.validate():
            raise ValueError("Please check your config.")

        return sherpa_onnx.OfflineSourceSeparation(_cf)

    start = time.time()
    try:
        sp = create_offline_source_separation()
        _separate_chunks(sp, input_file, vocal_file, instr_file, vocals_first=False, logs_file=logs_file)
        elapsed_seconds = time.time() - start
        _write_log(logs_file, f" use time:{elapsed_seconds:.3f}s")
        logger.debug(f'分离背景声和人声成功[{uvr_models}],耗时 {elapsed_seconds:.3f}s')
        return True, None
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f"人声背景声分离失败{e}:{msg}", exc_info=True)
        return False, f'{e}{msg}'


def vocal_bgm_spleeter(*, input_file, vocal_file, instr_file, logs_file=None,**kw):
    import sherpa_onnx

    def create_offline_source_separation():
        vocals = f"{ROOT_DIR}/models/onnx/vocals.fp16.onnx"
        accompaniment = f"{ROOT_DIR}/models/onnx/accompaniment.fp16.onnx"
        config = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                    vocals=vocals,
                    accompaniment=accompaniment,
                ),
                num_threads=int(settings.get('noise_separate_nums', 4)),
                debug=False,
                provider="cpu",
            )
        )
        if not config.validate():
            raise ValueError("Please check your config.")

        return sherpa_onnx.OfflineSourceSeparation(config)

    start = time.time()
    try:
        sp = create_offline_source_separation()
        _separate_chunks(sp, input_file, vocal_file, instr_file, vocals_first=True, logs_file=logs_file)
        elapsed_seconds = time.time() - start
        _write_log(logs_file, f" use time:{elapsed_seconds:.3f}s")
        logger.debug(f"分离背景声和人声成功[spleeter],耗时: {elapsed_seconds:.3f}s")
        return True, None
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f"人声背景声分离失败{e}:{msg}", exc_info=True)
        return False, f'{e}{msg}'


def _frames(stem):
    import numpy as np
    data = np.asarray(stem.data if hasattr(stem, "data") else stem)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.shape[0] <= 8 and data.shape[-1] > data.shape[0]:
        data = np.transpose(data)
    return np.ascontiguousarray(data, dtype=np.float32)


def _crossfade(tail, head):
    import numpy as np
    n = min(len(tail), len(head))
    if n <= 0:
        return head
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    return tail[:n] * ramp + head[:n] * (1.0 - ramp)


def _separate_chunks(separator, input_file, vocal_file, instr_file, vocals_first, logs_file=None):
    import numpy as np
    import soundfile as sf

    info = sf.info(input_file)
    sample_rate = info.samplerate
    chunk = max(int(_CHUNK_SECONDS * sample_rate), 1)
    overlap = min(int(_OVERLAP_SECONDS * sample_rate), chunk // 4)
    total = info.frames
    hop = max(chunk - overlap, 1)
    pending_vocal = None
    pending_music = None
    done = 0

    with sf.SoundFile(input_file) as source, \
            sf.SoundFile(vocal_file, "w", samplerate=sample_rate, channels=info.channels, subtype="PCM_16") as vocals_out, \
            sf.SoundFile(instr_file, "w", samplerate=sample_rate, channels=info.channels, subtype="PCM_16") as music_out:
        while done < total:
            source.seek(done)
            block = source.read(chunk, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            got = len(block)
            last = done + got >= total
            samples = np.ascontiguousarray(np.transpose(block))
            output = separator.process(sample_rate=sample_rate, samples=samples)
            stems = output.stems
            vocal = _frames(stems[0] if vocals_first else stems[1])
            music = _frames(stems[1] if vocals_first else stems[0])
            # The model drops a few samples at the end of each chunk.
            if len(vocal) < got:
                vocal = np.pad(vocal, ((0, got - len(vocal)), (0, 0)))
            if len(music) < got:
                music = np.pad(music, ((0, got - len(music)), (0, 0)))
            vocal = vocal[:got]
            music = music[:got]
            keep = got if last or got <= overlap else got - overlap
            if pending_vocal is not None and got >= overlap:
                vocal[:overlap] = _crossfade(pending_vocal, vocal[:overlap])
                music[:overlap] = _crossfade(pending_music, music[:overlap])
            vocals_out.write(vocal[:keep])
            music_out.write(music[:keep])
            if not last and got > overlap:
                pending_vocal = vocal[-overlap:].copy()
                pending_music = music[-overlap:].copy()
                done += hop
            else:
                done += got
            logger.debug(f"separation {min(done, total) / sample_rate:.0f}s / {total / sample_rate:.0f}s")
            _write_log(logs_file, f"separation {min(done, total) / sample_rate:.0f}s / {total / sample_rate:.0f}s")
