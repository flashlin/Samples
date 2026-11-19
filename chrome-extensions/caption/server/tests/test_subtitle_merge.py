import pytest

from caption_server.srt_codec import Cue, parse_srt, serialize_srt
from caption_server.subtitle_merge import append_offset, covered_seconds, merge_on_timeline, shift, union_intervals


def texts(cues: list[Cue]) -> list[str]:
    return [cue.text for cue in cues]


def test_srt_round_trip_with_bilingual_cue_and_bom() -> None:
    cues = [Cue(0.0, 2.5, "第一句"), Cue(3725.1, 3727.25, "Hello there\n你好")]
    content = "﻿" + serialize_srt(cues)
    assert parse_srt(content) == cues
    assert serialize_srt(cues).startswith("1\n00:00:00,000 --> 00:00:02,500\n第一句\n")
    assert parse_srt("") == []


def test_new_recording_replaces_only_its_own_time_range() -> None:
    old = [Cue(0, 10, "A"), Cue(10, 20, "B"), Cue(20, 30, "C"), Cue(30, 40, "D"), Cue(40, 50, "E")]
    new = [Cue(20, 30, "c'"), Cue(30, 40, "d'")]
    assert texts(merge_on_timeline(old, new)) == ["A", "B", "c'", "d'", "E"]


def test_old_cue_mostly_covered_is_removed() -> None:
    merged = merge_on_timeline([Cue(15, 25, "B")], [Cue(20, 30, "new")])
    assert texts(merged) == ["new"]


def test_old_cue_slightly_covered_is_kept_and_trimmed() -> None:
    merged = merge_on_timeline([Cue(12, 22, "B")], [Cue(20, 30, "new")])
    assert merged == [Cue(12, 20, "B"), Cue(20, 30, "new")]


def test_old_cue_trimmed_from_the_front_keeps_the_later_part() -> None:
    merged = merge_on_timeline([Cue(28, 40, "D")], [Cue(20, 30, "new")])
    assert merged == [Cue(20, 30, "new"), Cue(30, 40, "D")]


def test_silence_in_the_new_recording_keeps_old_captions() -> None:
    old = [Cue(30, 33, "C")]
    new = [Cue(20, 29, "x"), Cue(34, 40, "y")]
    assert texts(merge_on_timeline(old, new)) == ["x", "C", "y"]


def test_no_new_cues_keeps_everything() -> None:
    old = [Cue(0, 5, "A")]
    assert merge_on_timeline(old, []) == old


def test_union_and_covered_seconds() -> None:
    assert union_intervals([(40, 100), (0, 20), (18, 42), (5, 5)]) == [(0, 100)]
    assert covered_seconds([(0, 20), (40, 100)]) == pytest.approx(80)


def test_append_places_new_recording_after_the_old_file() -> None:
    old = [Cue(0, 2700, "first meeting")]
    offset = append_offset(old, [(0, 2710)])
    assert offset == pytest.approx(2712)
    cues, coverage = shift([Cue(0, 5, "second")], [(0, 1800)], offset)
    assert cues == [Cue(2712, 2717, "second")]
    assert coverage == [(2712, 4512)]
    assert append_offset([], []) == 0.0
