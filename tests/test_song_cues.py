from videotrans.task._song_cues import is_song_marker, without_song_cues


def test_only_the_marker_counts_as_a_song():
    assert is_song_marker("[SONG]")
    assert is_song_marker(" song ")
    assert not is_song_marker("Åh nej")
    assert not is_song_marker("")


def test_song_pairs_are_removed_together():
    source = [{"text": "Oh, oh, oh"}, {"text": "Oh no"}]
    target = [{"text": "[SONG]"}, {"text": "Åh nej"}]
    kept_s, kept_t, dropped = without_song_cues(source, target)
    assert [line["text"] for line in kept_s] == ["Oh no"]
    assert [line["text"] for line in kept_t] == ["Åh nej"]
    assert dropped == [0]


def test_a_short_translation_does_not_delete_lines():
    source = [{"text": "Hello"}, {"text": "There"}]
    target = [{"text": "Hej"}]
    kept_s, kept_t, dropped = without_song_cues(source, target)
    assert kept_s == source
    assert kept_t == target
    assert dropped == []
