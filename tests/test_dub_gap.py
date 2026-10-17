from videotrans.task._dub_gap import (
    centered_start_ms,
    early_start_slots,
    move_long_lines_earlier,
    silence_before_centered,
)


def test_a_shorter_line_starts_later_so_it_sits_in_the_middle():
    # Cue 10.000–12.000, dub 1.000 → 500 ms of silence on each side.
    assert centered_start_ms(10000, 12000, 1000) == 10500


def test_a_longer_line_starts_half_the_overflow_earlier():
    # Cue 10.000–12.000, dub 4.000 → start at 9.000, end at 13.000.
    assert centered_start_ms(10000, 12000, 4000) == 9000


def test_a_line_at_the_beginning_cannot_start_before_zero():
    assert centered_start_ms(0, 2000, 4000) == 0


def test_the_next_line_waits_only_for_the_half_that_still_overlaps():
    # Previous centered dub ends at 13.000. Next cue wants to start at 12.500.
    assert silence_before_centered(13000, 12500, 14000, 1500) == 0
    assert centered_start_ms(12500, 14000, 1500) == 12500


def test_a_line_after_a_gap_gets_silence_up_to_its_center():
    assert silence_before_centered(0, 10000, 12000, 1000) == 10500


def test_a_cramped_line_starts_earlier_in_the_previous_gap():
    # Previous cue has 3260 ms unused. This dub needs 214 ms more to stay at 1.3x.
    widths, borrows = early_start_slots([5280, 240], [2020, 590])
    assert borrows == [(1, 214)]
    assert widths == [5066, 454]
    assert sum(widths) == 5280 + 240


def test_a_line_does_not_start_more_than_two_seconds_earlier():
    widths, borrows = early_start_slots([20000, 1260], [1000, 7030])
    assert borrows == [(1, 2000)]
    assert widths == [18000, 3260]


def test_a_full_previous_line_is_left_alone():
    widths, borrows = early_start_slots([1300, 1300], [7120, 7120])
    assert borrows == []
    assert widths == [1300, 1300]


def test_move_long_lines_earlier_keeps_the_timeline_end():
    items = [
        {"start_time": 0, "end_time": 5280, "dubb_time": 2020},
        {"start_time": 5280, "end_time": 5520, "dubb_time": 590},
    ]
    move_long_lines_earlier(items)
    assert items[1]["start_time"] == 5280 - 214
    assert items[1]["end_time"] == 5520
    assert items[0]["end_time"] == items[1]["start_time"]
