# One batch of ChatterBox lines in its own process, so the model and GPU
# workspace are released when the batch ends.
import inspect
import json
from pathlib import Path

import soundfile as sf

from videotrans.configure.config import logger, settings
from videotrans.process._utils import convert_to_wav
from videotrans.tts._designations import speakable_text
from videotrans.util.gpus import mps_or_cpu
from videotrans.util.help_misc import vail_file

# Chatterbox drops the last five text columns when it looks for a repeated
# line. A one-word line is narrower than that, so the slice is empty and
# ``max(dim=1)`` crashes. Compare how many text tokens went in with how many
# have been covered: one in and one covered is a pass.
_ALIGNMENT_OLD = (
    "alignment_repetition = self.complete and "
    "(A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)"
)
_ALIGNMENT_NEW = (
    "alignment_repetition = repetition_detected(\n"
    "            self.complete, S, int(self.text_position) + 1, A, self.completed_at)"
)


def patch_alignment_source(text):
    if _ALIGNMENT_OLD not in text:
        return text
    return text.replace(_ALIGNMENT_OLD, _ALIGNMENT_NEW, 1)


def repetition_detected(complete, text_tokens, covered_tokens, alignment, completed_at):
    """True when speech starts a second lap over text that was already spoken.

    One text token covered once passes. Audio frames are not counted one-to-one
    against text tokens, because one word is many frames.
    """
    if not complete or completed_at is None:
        return False
    text_tokens = int(text_tokens)
    covered_tokens = int(covered_tokens)
    if int(alignment.shape[1]) <= 5:
        return covered_tokens > text_tokens
    earlier = alignment[completed_at:, :-5]
    if int(earlier.shape[-1]) == 0:
        return covered_tokens > text_tokens
    return bool(earlier.max(dim=1).values.sum() > 5)


def guard_alignment_step(original):
    """Catch the empty-slice ``max(dim=1)`` without writing the installed package.

    The venv file is read-only in the container, so a source rewrite never lands.
    """

    def step(self, logits, next_token=None):
        try:
            return original(self, logits, next_token)
        except IndexError:
            return logits

    step._short_guard = True
    return step


def _patch_short_alignment():
    import textwrap

    import chatterbox.models.t3.inference.alignment_stream_analyzer as mod

    if getattr(mod.AlignmentStreamAnalyzer.step, "_short_guard", False):
        return
    source = textwrap.dedent(inspect.getsource(mod.AlignmentStreamAnalyzer.step))
    patched = patch_alignment_source(source)
    if patched == source:
        mod.AlignmentStreamAnalyzer.step = guard_alignment_step(
            mod.AlignmentStreamAnalyzer.step
        )
        return
    mod.repetition_detected = repetition_detected
    namespace = dict(mod.__dict__)
    exec(patched, namespace)  # noqa: S102
    step = namespace["step"]
    step._short_guard = True
    mod.AlignmentStreamAnalyzer.step = step


def clip_prompt_samples(samples, sample_rate, *, max_seconds=8.0):
    """Mono frames, capped. A 12s stereo reference is what the crashed line used."""
    frames = []
    for frame in samples:
        if isinstance(frame, (int, float)):
            frames.append(float(frame))
        else:
            frames.append(sum(frame) / len(frame))
    limit = max(int(sample_rate * max_seconds), 1)
    return frames[:limit]


def _prepare_prompt(ref_wav):
    if not ref_wav or not vail_file(ref_wav):
        return None
    dest = Path(ref_wav).with_name(Path(ref_wav).stem + "-prompt.wav")
    if vail_file(dest):
        return str(dest)
    audio, sample_rate = sf.read(ref_wav, always_2d=True)
    clipped = clip_prompt_samples(audio.tolist(), sample_rate)
    if len(clipped) < int(sample_rate * 0.3):
        return ref_wav
    sf.write(dest, clipped, sample_rate)
    return str(dest)


def _write_silence(item):
    duration = max(int(item.get("end_time") or 0) - int(item.get("start_time") or 0), 200) / 1000
    raw = item["filename"] + "-24k.wav"
    sf.write(raw, [0.0] * max(int(24000 * duration), 1), 24000)
    convert_to_wav(raw, item["filename"])


def chatterbox_fun(*, queue_tts_file, language, is_cuda=False, local_dir="", cfg_weight=0.5, exaggeration=0.5, **kw):
    _patch_short_alignment()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    queue = json.loads(Path(queue_tts_file).read_text(encoding="utf-8"))
    device = settings.get("device_name", "auto")
    if device == "auto":
        device = "cuda" if is_cuda else mps_or_cpu()
    logger.debug(f"Chatterbox batch {len(queue)} lines on {device}")

    load_kwargs = {"device": device}
    if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_local).parameters:
        load_kwargs["t3_model"] = "v3"
    model = ChatterboxMultilingualTTS.from_local(local_dir, **load_kwargs)
    lang = (language or "en").split("-")[0]
    ok = 0
    for i, item in enumerate(queue, start=1):
        if vail_file(item.get("filename")):
            ok += 1
            continue
        text = speakable_text((item.get("speak_text") or item.get("text") or "").strip())
        if not text:
            continue
        ref_wav = _prepare_prompt(item.get("ref_wav"))
        logger.debug(f"Chatterbox {i}/{len(queue)} {text[:40]}")
        try:
            wav = model.generate(
                text,
                exaggeration=float(exaggeration),
                cfg_weight=float(cfg_weight),
                language_id=lang,
                audio_prompt_path=ref_wav,
            )
        except Exception as exc:
            logger.exception(f"Chatterbox skipped line {i}: {text[:40]}: {exc}")
            _write_silence(item)
            ok += 1
            continue
        wav = wav.detach().cpu()
        wav_np = wav.transpose(0, 1).numpy() if wav.ndim == 2 else wav.numpy()
        raw = item["filename"] + "-24k.wav"
        sf.write(raw, wav_np, model.sr, format="wav")
        del wav, wav_np
        convert_to_wav(raw, item["filename"])
        ok += 1
    del model
    return True, None
