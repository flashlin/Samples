import re
from dataclasses import dataclass

MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
_TIMING = re.compile(r"^(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * SECONDS_PER_HOUR + int(minutes) * SECONDS_PER_MINUTE + int(seconds) + int(millis) / MILLISECONDS_PER_SECOND


def _timestamp(value: float) -> str:
    total = max(0, round(value * MILLISECONDS_PER_SECOND))
    millis = total % MILLISECONDS_PER_SECOND
    whole = total // MILLISECONDS_PER_SECOND
    return f"{whole // SECONDS_PER_HOUR:02d}:{whole % SECONDS_PER_HOUR // SECONDS_PER_MINUTE:02d}:{whole % SECONDS_PER_MINUTE:02d},{millis:03d}"


def parse_srt(content: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in re.split(r"\r?\n\s*\r?\n", content.lstrip("﻿").strip()):
        lines = block.splitlines()
        for index, line in enumerate(lines):
            match = _TIMING.match(line.strip())
            if match:
                text = "\n".join(lines[index + 1 :]).strip()
                if text:
                    cues.append(Cue(_seconds(*match.groups()[:4]), _seconds(*match.groups()[4:]), text))
                break
    return sorted(cues, key=lambda cue: (cue.start, cue.end))


def serialize_srt(cues: list[Cue]) -> str:
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    return "\n".join(f"{index}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.text}\n" for index, cue in enumerate(ordered, start=1))
