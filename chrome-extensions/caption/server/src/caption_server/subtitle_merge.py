from dataclasses import dataclass
from typing import Literal

from .srt_codec import Cue

Placement = Literal["timeline", "append"]
Interval = tuple[float, float]

REPLACE_OVERLAP_RATIO = 0.5
APPEND_GAP_SECONDS = 2.0


def union_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted(interval for interval in intervals if interval[1] > interval[0]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def covered_seconds(intervals: list[Interval]) -> float:
    return sum(end - start for start, end in union_intervals(intervals))


def _overlap(cue: Cue, spans: list[Interval]) -> float:
    return sum(max(0.0, min(cue.end, end) - max(cue.start, start)) for start, end in spans)


def _trim_outside(cue: Cue, spans: list[Interval]) -> Cue | None:
    pieces: list[Interval] = [(cue.start, cue.end)]
    for start, end in spans:
        next_pieces: list[Interval] = []
        for piece_start, piece_end in pieces:
            if end <= piece_start or start >= piece_end:
                next_pieces.append((piece_start, piece_end))
                continue
            if start > piece_start:
                next_pieces.append((piece_start, start))
            if end < piece_end:
                next_pieces.append((end, piece_end))
        pieces = next_pieces
    if not pieces:
        return None
    start, end = max(pieces, key=lambda piece: piece[1] - piece[0])
    return Cue(start, end, cue.text)


def merge_on_timeline(old: list[Cue], new: list[Cue]) -> list[Cue]:
    spans = union_intervals([(cue.start, cue.end) for cue in new])
    kept: list[Cue] = []
    for cue in old:
        duration = cue.end - cue.start
        overlap = _overlap(cue, spans)
        if overlap <= 0:
            kept.append(cue)
        elif duration > 0 and overlap / duration < REPLACE_OVERLAP_RATIO:
            trimmed = _trim_outside(cue, spans)
            if trimmed:
                kept.append(trimmed)
    return sorted([*kept, *new], key=lambda cue: (cue.start, cue.end))


def append_offset(old: list[Cue], old_coverage: list[Interval]) -> float:
    ends = [cue.end for cue in old] + [end for _, end in old_coverage]
    return max(ends) + APPEND_GAP_SECONDS if ends else 0.0


def shift(cues: list[Cue], coverage: list[Interval], offset: float) -> tuple[list[Cue], list[Interval]]:
    return [Cue(cue.start + offset, cue.end + offset, cue.text) for cue in cues], [(start + offset, end + offset) for start, end in coverage]


@dataclass(frozen=True)
class Recording:
    cues: list[Cue]
    coverage: list[Interval]


def combine_recordings(base: Recording | None, new: Recording, placement: Placement) -> Recording:
    if base is None:
        return Recording(sorted(new.cues, key=lambda cue: (cue.start, cue.end)), union_intervals(new.coverage))
    cues, coverage = new.cues, new.coverage
    if placement == "append":
        cues, coverage = shift(cues, coverage, append_offset(base.cues, base.coverage))
    return Recording(merge_on_timeline(base.cues, cues), union_intervals([*base.coverage, *coverage]))
