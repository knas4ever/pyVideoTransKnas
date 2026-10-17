from videotrans.task._keep_effects import speech_mute_filter, speech_mute_ranges


def test_speech_ranges_merge_and_leave_a_gap_for_r2d2():
    lines = [
        {"start_time": 1000, "end_time": 2000},
        {"start_time": 2100, "end_time": 2500},
        {"start_time": 5000, "end_time": 5600},
    ]
    ranges = speech_mute_ranges(lines, pad_ms=100)
    assert ranges == [(900, 2600), (4900, 5700)]


def test_mute_filter_silences_only_the_speech():
    lines = [{"start_time": 1000, "end_time": 2000}]
    assert speech_mute_filter(lines, pad_ms=0) == "volume=0:enable='between(t,1.000,2.000)'"
    assert speech_mute_filter([]) is None
