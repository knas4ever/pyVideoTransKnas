from videotrans.process.chatterbox_tts import (
    clip_prompt_samples,
    guard_alignment_step,
    patch_alignment_source,
    repetition_detected,
)


class _Shape:
    def __init__(self, columns):
        self.shape = (3, columns)


def test_one_token_in_and_one_covered_passes():
    alignment = _Shape(1)
    assert repetition_detected(True, 1, 1, alignment, 0) is False


def test_a_second_lap_over_one_token_is_repetition():
    alignment = _Shape(1)
    assert repetition_detected(True, 1, 2, alignment, 0) is True


def test_short_alignment_compares_token_counts():
    source = (
        "        alignment_repetition = self.complete and "
        "(A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)\n"
    )
    patched = patch_alignment_source(source)
    assert "repetition_detected" in patched
    assert "].max(dim=1)" not in patched


def test_guard_returns_logits_when_alignment_is_too_narrow():
    def original(self, logits, next_token=None):
        raise IndexError("max(): Expected reduction dim 1 to have non-zero size.")

    step = guard_alignment_step(original)
    assert step(None, "logits") == "logits"


def test_long_stereo_prompt_is_clipped_to_eight_seconds():
    stereo = [(1.0, -1.0)] * 16000 * 12
    clipped = clip_prompt_samples(stereo, 16000)
    assert len(clipped) == 16000 * 8
    assert clipped[0] == 0.0


def test_patch_is_idempotent():
    source = (
        "        alignment_repetition = self.complete and "
        "(A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)\n"
    )
    once = patch_alignment_source(source)
    assert patch_alignment_source(once) == once
