# 语音识别，新进程执行
# 返回元组
# 失败：第一个值为False，则为失败，第二个值存储失败原因
# 成功，第一个值存在需要的返回值，不需要时返回True，第二个值为None
import re, json, traceback
from pathlib import Path
from videotrans.configure.config import logger
from videotrans.process._stt_utils import _resegment, _write_log

# cut_audio puts this much silence in front of every clip.
CLIP_PAD_S = 0.4
_ATTACHED = set(".?!。？！,;:，；：、")



def group_parakeet_tokens(tokens):
    """Join Parakeet tokens into words and keep the period on the word it closes."""
    words = []
    buf = ""
    start = None
    end = None

    def flush():
        nonlocal buf, start, end
        text = buf.strip()
        if text and start is not None:
            words.append({
                "word": text,
                "start": float(start),
                "end": float(end if end is not None else start),
            })
        buf = ""
        start = None
        end = None

    for tok in tokens or []:
        piece = str(tok.get("token", ""))
        if not piece:
            continue
        mark = piece.strip()
        if mark in _ATTACHED:
            if buf:
                buf += mark
                end = tok.get("end", end)
                flush()
            elif words:
                words[-1]["word"] += mark
                words[-1]["end"] = float(tok.get("end", words[-1]["end"]))
            continue
        if buf and piece[:1].isspace():
            flush()
        if not buf:
            start = tok.get("start", 0)
        buf += piece
        end = tok.get("end", start)
    flush()
    return words


def place_clip_words(words, chunk_start_s, pad_s=CLIP_PAD_S):
    """Move clip-local word times onto the film clock.

    The clip file starts with silence. Subtract that pad only when the first
    word actually begins after it, so a model that already starts at zero
    keeps its spacing.
    """
    if not words:
        return []
    first = min(float(word["start"]) for word in words)
    shift = pad_s if first >= pad_s * 0.5 else 0.0
    placed = []
    origin = float(chunk_start_s)
    for word in words:
        start = round(origin + max(0.0, float(word["start"]) - shift), 3)
        end = round(origin + max(0.0, float(word["end"]) - shift), 3)
        if end < start:
            end = start
        placed.append({"word": word["word"], "start": start, "end": end})
    return placed


def _is_parakeet(local_dir, model_name):
    return "parakeet" in f"{local_dir} {model_name}".lower()


def _decode_parakeet_clip(processor, output):
    decoded = processor.decode(
        output.sequences,
        durations=getattr(output, "durations", None),
        skip_special_tokens=True,
    )
    timestamps = []
    if isinstance(decoded, tuple):
        text_list, timestamps = decoded
    else:
        text_list = decoded
    if isinstance(text_list, str):
        text = text_list.strip()
    else:
        text = str(text_list[0]).strip() if text_list else ""
    token_times = []
    if timestamps:
        first = timestamps[0]
        token_times = first if isinstance(first, list) else timestamps
    return text, token_times


def _parakeet_segments(raws, local_dir, logs_file, device_name):
    from transformers import AutoModelForTDT, AutoProcessor
    from transformers.audio_utils import load_audio

    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForTDT.from_pretrained(local_dir, device_map=device_name or "auto")
    msg = f"running on {model.device}"
    _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
    logger.debug(f"huggingface_asr渠道使用模型: {local_dir},{msg}, word timestamps")
    rate = processor.feature_extractor.sampling_rate
    segments = []
    total = len(raws)
    for i, it in enumerate(raws):
        _write_log(logs_file, json.dumps({"type": "logs", "text": f"Subtitles {i + 1}/{total}..."}))
        audio = load_audio(it["filename"], sampling_rate=rate)
        inputs = processor(audio, sampling_rate=rate)
        inputs.to(model.device, dtype=model.dtype)
        output = model.generate(**inputs, return_dict_in_generate=True)
        text, token_times = _decode_parakeet_clip(processor, output)
        words = place_clip_words(group_parakeet_tokens(token_times), it["start_time"] / 1000.0)
        if text:
            _write_log(logs_file, json.dumps({"type": "subtitles", "text": f"[{i}] {text}\n"}))
        if not words:
            if text:
                raws[i]["text"] = text
            continue
        segments.append({
            "text": text,
            "start": it["start_time"] / 1000.0,
            "end": it["end_time"] / 1000.0,
            "words": words,
        })
    return segments


def pipe_asr(
        prompt=None,
        cut_audio_list=None,
        detect_language=None,
        logs_file=None,
        local_dir=None,
        **kw
):
    from transformers import pipeline

    def inputs_generator():
        for item in raws:
            yield item['filename']

    detect_language = 'tl' if detect_language == 'fil' else detect_language

    try:
        if cut_audio_list and isinstance(cut_audio_list, str):
            cut_audio_list = [item for item in
                                             json.loads(Path(cut_audio_list).read_text(encoding='utf-8'))]
        raws = cut_audio_list
        if _is_parakeet(local_dir, kw.get("model_name")):
            try:
                segments = _parakeet_segments(
                    raws, local_dir, logs_file, kw.get("device_name", "auto")
                )
                if segments:
                    language = detect_language.split("-")[0] if detect_language and detect_language != "auto" else "en"
                    return _resegment(segments, language, logs_file=logs_file), None
                if any(str(item.get("text", "")).strip() for item in raws):
                    return raws, None
            except Exception:
                logger.exception("Parakeet word timestamps failed, falling back to plain text")
        p = pipeline(
            task="automatic-speech-recognition",
            model=local_dir,
            batch_size=4,
            device_map=kw.get('device_name','auto'),
            dtype='auto'
        )
        msg = f"running on {p.model.device}"
        _write_log(logs_file, json.dumps({"type": "logs", "text": msg}))
        logger.debug(f'huggingface_asr渠道使用模型: {local_dir},{msg}')

        generate_kwargs = {}
        # 仅 openai官方模型加 generate_kwargs 参数，其他微调可能不兼容，暂不加
        if "--openai--whisper" in local_dir:
            lang = detect_language.split('-')[0] if detect_language != 'auto' else None
            generate_kwargs["task"] = "transcribe"
            if lang:
                generate_kwargs["language"] = lang
            if prompt:
                if hasattr(p.tokenizer, "get_prompt_ids"):
                    prompt_ids = p.tokenizer.get_prompt_ids(prompt, return_tensors="pt")
                else:
                    prompt_ids = p.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

                prompt_ids = prompt_ids.to(p.model.device)

                generate_kwargs["prompt_ids"] = prompt_ids

        results_iterator = p(
            inputs_generator(),
            generate_kwargs=generate_kwargs
        )

        total = len(raws)

        for i, (it, res) in enumerate(zip(raws, results_iterator)):
            _write_log(logs_file, json.dumps({"type": "logs", "text": f"Subtitles {i + 1}/{total}..."}))
            text = res.get('text', '')
            if text:
                cleaned_text = re.sub(r'<unk>|</unk>', '', text).strip()
                raws[i]['text'] = cleaned_text

                _write_log(logs_file, json.dumps({"type": "subtitles", "text": f'[{i}] {cleaned_text}\n'}))

        return raws, None
    except Exception as e:
        msg = traceback.format_exc()
        return False, f'{e}:{msg}'
