# -*- coding: utf-8 -*-
import traceback, time, json
from videotrans.configure.config import ROOT_DIR, logger, app_cfg
from pathlib import Path
from collections import defaultdict

"""

subtitles:List[List[start:int,end:int]]=[[0, 4580], [4580, 7860], [8260, 10520], [10520, 11880], [12200, 13260], [14020, 14140], [14460, 15660], [15660, 19640], [19640, 24040], [24040, 24880], [25100, 28600], [28720, 29260], [29260, 33220], [33220, 36540], [36540, 39460], [39460, 43420], [43420, 46400], [46400, 49420], [49420, 53840], [53840, 56480], [56480, 57020]]

diarizations:List[List[List[start:int,end:int],spk:str]]=[[[30, 41037], 'spk0'], [[30237, 30591], 'spk1'], [[41442, 57405], 'spk2']]

retun output=['spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk0', 'spk2', 'spk2', 'spk2', 'spk2', 'spk2', 'spk2']
"""


def _assign_speakers(subtitles, diarizations):
    # ----------------- 1. 预处理 diarizations -----------------
    clean_diars = []
    for dia in diarizations:
        if len(dia) == 2 and len(dia[0]) == 2 and dia[0][0] < dia[0][1]:
            clean_diars.append((dia[0][0], dia[0][1], dia[1]))
    clean_diars.sort(key=lambda x: x[0])  # 按开始时间排序

    # ----------------- 2. 预处理 subtitles（保留原顺序） -----------------
    indexed_subs = []
    for idx, sub in enumerate(subtitles):
        if len(sub) == 2 and sub[0] < sub[1]:
            indexed_subs.append((idx, sub[0], sub[1]))
        else:
            indexed_subs.append((idx, None, None))

    valid_subs = [s for s in indexed_subs if s[1] is not None]
    valid_subs.sort(key=lambda x: x[1])  # 只对有效片段按开始时间排序

    # 输出数组，默认全是 "spk0"
    output = ["spk0"] * len(subtitles)

    # ----------------- 3. 扫描线分配说话人 -----------------
    d_ptr = 0
    active = []  # 存储 (d_start, d_end, speaker)
    total_diars = len(clean_diars)

    for orig_idx, s_start, s_end in valid_subs:
        duration = s_end - s_start

        # 将开始时间 < VAD 结束时间 的 diar 加入窗口
        while d_ptr < total_diars and clean_diars[d_ptr][0] < s_end:
            active.append(clean_diars[d_ptr])
            d_ptr += 1

        # 移除窗口中已经结束的 diar（结束时间 <= VAD 开始时间）
        # 因为 active 很小，重建列表完全没问题
        active = [d for d in active if d[1] > s_start]

        # 计算重叠
        overlaps = defaultdict(int)
        for d_start, d_end, spk in active:
            o_start = max(s_start, d_start)
            o_end = min(s_end, d_end)
            overlaps[spk] += (o_end - o_start)

        if not overlaps:
            continue  # 保持 "spk0"

        num_speakers = len(overlaps)
        best_spk = max(overlaps, key=overlaps.get)
        max_overlap = overlaps[best_spk]

        if num_speakers > 1:
            output[orig_idx] = best_spk
        else:  # num_speakers == 1
            if max_overlap > 0.2 * duration:
                output[orig_idx] = best_spk
            # 否则保持 "spk0"
    return output


def _map_speakers(diarizations):
    speaker_list = sorted(list(set(d['speaker'] for d in diarizations)))
    spk_map = {spk: f'spk{i}' for i, spk in enumerate(speaker_list)}
    for d in diarizations:
        d['speaker'] = spk_map.get(d['speaker'], 'spk0')
    logger.debug(f'原始说话人排序后：{speaker_list=}')
    logger.debug(f'映射为新说话人标识：{spk_map=}')
    return diarizations


def _normalize_diarizations(raw_output):
    """将原始说话人分离结果标准化为统一格式"""
    output = []
    speaker_list = set()
    for item in raw_output:
        speaker_list.add(item['speaker'])
        output.append({
            'times': [int(item['start'] * 1000), int(item['end'] * 1000)],
            'speaker': item['speaker']
        })
    speaker_list = sorted(list(speaker_list))
    spk_map = {spk: f'spk{i}' for i, spk in enumerate(speaker_list)}
    for d in output:
        d['speaker'] = spk_map.get(d['speaker'], 'spk0')
    logger.debug(f'原始说话人排序后：{speaker_list=}')
    logger.debug(f'映射为新说话人标识：{spk_map=}')
    return output


def _diarize_and_write(subtitles_file, diarizations, speak_file):
    """通用：读取字幕文件，将说话人分离结果分配到字幕，写入结果"""
    subtitles = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
    diar_list = [[d['times'], d['speaker']] for d in diarizations]
    output = _assign_speakers(subtitles, diar_list)
    if output:
        Path(speak_file).write_text(json.dumps(output), encoding='utf-8')
        return True, None
    return False, "0 speakers"


def cam_speakers(*, input_file, subtitles_file: str, speak_file: str, num_speakers=-1, is_cuda=False,  device_index=0,**kw):
    from modelscope.pipelines import pipeline
    device=kw.get('device_name','auto')
    if device=='auto':
        device = f"cuda:{device_index}" if is_cuda else "cpu"
    _st = time.time()
    logger.debug(f'开始说话人分离:使用阿里cam++模型')

    try:
        subtitles = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
        ans = pipeline(
            task='speaker-diarization',
            model=f'{ROOT_DIR}/models/speech_campplus_speaker-diarization_common',
            disable_update=True,
            disable_progress_bar=True,
            disable_log=True,
            device=device
        )
        result = ans(input_file, oracle_num=num_speakers, ignore_errors=True) if num_speakers > 1 else ans(input_file,
                                                                                                           ignore_errors=True)
        logger.debug(f'说话人分离原始返回结果:{result=}')
        diarizations = _normalize_diarizations(
            [{'start': it[0], 'end': it[1], 'speaker': f'spk{int(it[2])}'} for it in result['text']]
        )
        diar_list = [[d['times'], d['speaker']] for d in diarizations]
        output = _assign_speakers(subtitles, diar_list)
        logger.debug(f'说话人分离成功结束,识别出 {len(set(output))} 个说话人,耗时:{int(time.time() - _st)}s')
        if output:
            Path(speak_file).write_text(json.dumps(output), encoding='utf-8')
            return True, None
        return False, "0 speakers"
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f'说话人分离失败{e}:{msg}', exc_info=True)
        return False, f'{e}{msg}'


# pyannote 3.4 依赖 huggingface_hub<1.0，高于会出现 use_auth_token 被废弃改为 token错误，而其他很多模块要求 huggingface_hub>1.0, 因此通过补丁允许 huggingface_hub仍接受 废弃的 use_auth_token
def _hook_hf():
    import os
    if app_cfg.proxy:
        os.environ['HTTPS_PROXY'] = app_cfg.proxy
        os.environ['HTTP_PROXY'] = app_cfg.proxy

    import huggingface_hub.file_download

    # 1. 备份原生的 hf_hub_download 方法
    _original_hf_hub_download = huggingface_hub.file_download.hf_hub_download

    # 2. 定义我们自己的包装（拦截）函数
    def patched_hf_hub_download(*args, **kwargs):
        # 拦截并处理 use_auth_token
        if "use_auth_token" in kwargs:
            # 取出 use_auth_token 并从 kwargs 中删除
            auth_token = kwargs.pop("use_auth_token")

            # 将其赋值给新版 hf_hub_download 认的 'token' 参数
            if "token" not in kwargs:
                kwargs["token"] = auth_token

        # 调用并返回原本的下载逻辑
        return _original_hf_hub_download(*args, **kwargs)

    # 3. 替换 huggingface_hub 中的方法为我们的包装函数
    huggingface_hub.hf_hub_download = patched_hf_hub_download
    huggingface_hub.file_download.hf_hub_download = patched_hf_hub_download


def pyannote_speakers(*, input_file, subtitles_file: str, speak_file: str, num_speakers=-1, is_cuda=False,
                     device_index=0,**kw):
    _hook_hf()
    import torch, pyannote.audio, torchaudio
    torch.serialization.add_safe_globals([
        torch.torch_version.TorchVersion,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution
    ])
    from pyannote.audio import Pipeline

    def _get_diariz():
        pipeline = Pipeline.from_pretrained(f"{ROOT_DIR}/models/models--pyannote--speaker-diarization-3.1/config.yaml")
        _device_name=kw.get('device_name','auto')
        if _device_name!='auto':
            pipeline.to(torch.device(_device_name))
        elif is_cuda:
            pipeline.to(torch.device(f"cuda:{device_index}"))

        waveform, sample_rate = torchaudio.load(input_file)
        if num_speakers > 0:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
        else:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        raw_output = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker = speaker.replace('SPEAKER_', '')
            raw_output.append({'start': turn.start, 'end': turn.end, 'speaker': f'spk{speaker}'})
        return _normalize_diarizations(raw_output)

    try:
        _st = time.time()
        logger.debug(f'开始说话人分离,使用 pyannote/speaker-diarization-3.1 模型')
        subtitles = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
        diarizations = _get_diariz()
        if not diarizations:
            return False, "Unkonw error"
        diar_list = [[d['times'], d['speaker']] for d in diarizations]
        output = _assign_speakers(subtitles, diar_list)
        logger.debug(f'说话人分离成功结束,识别出个 {len(set(output))} 说话人,耗时:{int(time.time() - _st)}s')
        if output:
            Path(speak_file).write_text(json.dumps(output), encoding='utf-8')
            return True, None
        return False, "0 speakers"
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f'说话人分离出错{e}:{msg}', exc_info=True)
        return False, f'{e}{msg}'


def built_speakers(*, input_file, subtitles_file: str, speak_file: str, num_speakers=-1, language="zh", **kw):
    import librosa
    import soundfile as sf

    def resample_audio(audio, sample_rate, target_sample_rate):
        if sample_rate != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            return audio, target_sample_rate
        return audio, sample_rate

    def init_speaker_diarization(language, num_speakers=-1, min_duration_off=0.5):
        import sherpa_onnx
        segmentation_model = f"{ROOT_DIR}/models/onnx/seg_model.onnx"
        embedding_extractor_model = (
            f"{ROOT_DIR}/models/onnx/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx" if language == 'zh' else f"{ROOT_DIR}/models/onnx/nemo_en_titanet_small.onnx"
        )
        if not Path(embedding_extractor_model).exists():
            raise RuntimeError('Not found speaker_diarization model')

        _cf = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=segmentation_model
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=embedding_extractor_model
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers, threshold=0.5
            ),
            min_duration_on=0.2,
            min_duration_off=min_duration_off,
        )
        if not _cf.validate():
            raise RuntimeError(
                "Please check your config and make sure all required files exist"
            )

        return sherpa_onnx.OfflineSpeakerDiarization(_cf), embedding_extractor_model

    def _progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
        return int(num_processed_chunk / num_total_chunks * 100 if num_total_chunks > 0 else 0)

    def _read_audio():
        audio, sample_rate = sf.read(input_file, dtype="float32", always_2d=True)
        return audio[:, 0], sample_rate

    def _run_model(min_duration_off):
        audio, sample_rate = _read_audio()
        sd, embedding_model = init_speaker_diarization(language, num_speakers, min_duration_off)
        audio, sample_rate = resample_audio(audio, sample_rate, sd.sample_rate)
        if sample_rate != sd.sample_rate:
            raise RuntimeError(
                f"Expected samples rate: {sd.sample_rate}, given: {sample_rate}"
            )
        result = sd.process(audio, callback=_progress_callback).sort_by_start_time()
        raw_output = []
        for r in result:
            raw_output.append({'start': r.start, 'end': r.end, 'speaker': f'spk{r.speaker}'})
        return _normalize_diarizations(raw_output), audio, sample_rate, embedding_model

    def _embed_span(extractor, audio, sample_rate, start_ms, end_ms):
        import numpy as np
        start = max(int(start_ms / 1000 * sample_rate), 0)
        end = min(int(end_ms / 1000 * sample_rate), len(audio))
        if end - start < int(0.3 * sample_rate):
            return None
        chunk = np.ascontiguousarray(audio[start:end])
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, chunk)
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return extractor.compute(stream)

    def _split_segments_on_silence(diarizations, audio, sample_rate, min_gap_s=0.18, min_piece_s=0.45):
        import numpy as np
        frame = int(0.02 * sample_rate)
        hop = max(int(0.01 * sample_rate), 1)
        if frame < 1 or len(audio) <= frame:
            return diarizations
        rms = np.array([
            float(np.sqrt(np.mean(audio[i:i + frame] ** 2)))
            for i in range(0, len(audio) - frame, hop)
        ])
        pieces = []
        for item in diarizations:
            start_ms, end_ms = item["times"]
            if end_ms - start_ms < 1200:
                pieces.append(item)
                continue
            i0 = int(start_ms / 1000 * sample_rate) // hop
            i1 = min(int(end_ms / 1000 * sample_rate) // hop, len(rms))
            region = rms[i0:i1]
            if len(region) < 5:
                pieces.append(item)
                continue
            threshold = max(float(np.percentile(region, 80)) * 0.18, 1e-4)
            gaps = []
            gap_start = None
            for index, value in enumerate(region):
                if value < threshold:
                    if gap_start is None:
                        gap_start = index
                elif gap_start is not None:
                    if (index - gap_start) * hop / sample_rate >= min_gap_s:
                        gaps.append((gap_start, index))
                    gap_start = None
            if not gaps:
                pieces.append(item)
                continue
            cuts = [start_ms]
            for gap_from, gap_to in gaps:
                cuts.append(start_ms + int(gap_from * hop / sample_rate * 1000))
                cuts.append(start_ms + int(gap_to * hop / sample_rate * 1000))
            cuts.append(end_ms)
            split = []
            for left, right in zip(cuts[0::2], cuts[1::2]):
                if right - left >= int(min_piece_s * 1000):
                    split.append({"times": [left, right], "speaker": item["speaker"]})
            pieces.extend(split if len(split) >= 2 else [item])
        if len(pieces) != len(diarizations):
            logger.debug(f'按静音切开长句：{[it["times"] for it in pieces]}')
        return pieces

    def _recluster(diarizations, audio, sample_rate, embedding_model):
        import sherpa_onnx
        from videotrans.task._speaker_split import cluster_by_cosine
        long_items = [item for item in diarizations if item['times'][1] - item['times'][0] >= 400]
        if len(long_items) < 2:
            return diarizations
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding_model)
        )
        vectors = []
        usable = []
        for item in long_items:
            vector = _embed_span(extractor, audio, sample_rate, item['times'][0], item['times'][1])
            if vector is None:
                continue
            usable.append(item)
            vectors.append(vector)
        if len(vectors) < 2:
            return diarizations
        labels = cluster_by_cosine(vectors, min_cosine=0.55)
        if len(set(labels)) < 2:
            return diarizations
        for item, label in zip(usable, labels):
            item['speaker'] = f'spk{label}'
        logger.debug(f'短音频二次分说话人：{[(it["times"], it["speaker"]) for it in diarizations]}')
        return diarizations

    try:
        _st = time.time()
        logger.debug(f'开始说话人分离,使用内置模型 {language=},{num_speakers=}')
        subtitles = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
        diarizations, audio, sample_rate, embedding_model = _run_model(0.5)
        if not diarizations:
            return False, 'Unknow error'
        if len({item['speaker'] for item in diarizations}) < 2:
            diarizations = _split_segments_on_silence(diarizations, audio, sample_rate)
            diarizations = _recluster(diarizations, audio, sample_rate, embedding_model)
        diar_list = [[d['times'], d['speaker']] for d in diarizations]
        output = _assign_speakers(subtitles, diar_list)
        logger.debug(f'说话人分离成功结束,识别出个 {len(set(output))} 说话人,耗时：{int(time.time() - _st)}s')
        if output:
            Path(speak_file).write_text(json.dumps(output), encoding='utf-8')
            spans = [
                {"start": item["times"][0], "end": item["times"][1], "speaker": item["speaker"]}
                for item in diarizations
            ]
            Path(speak_file).with_name("speaker_spans.json").write_text(
                json.dumps(spans), encoding="utf-8"
            )
            return True, None
        return False, "0 speakers"
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f'分离说话人失败:{e}', exc_info=True)
        return False, f'{e}{msg}'
