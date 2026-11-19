from dataclasses import dataclass
from typing import Literal

SubtitleStatus = Literal["partial", "complete"]
OverwriteRejection = Literal["empty-candidate", "stale-sequence", "confirmation-required"]


@dataclass(frozen=True)
class SubtitleProgress:
    status: SubtitleStatus
    covered_seconds: float
    session_id: str
    last_seq: int


@dataclass(frozen=True)
class OverwriteDecision:
    write: bool
    reason: OverwriteRejection | None = None


_ACCEPT = OverwriteDecision(write=True)


def decide_overwrite(existing: SubtitleProgress | None, candidate: SubtitleProgress, confirmed: bool) -> OverwriteDecision:
    if candidate.status == "partial" and candidate.covered_seconds <= 0:
        return OverwriteDecision(False, "empty-candidate")
    if existing is None:
        return _ACCEPT
    if existing.session_id == candidate.session_id:
        return _ACCEPT if candidate.last_seq > existing.last_seq else OverwriteDecision(False, "stale-sequence")
    return _ACCEPT if confirmed else OverwriteDecision(False, "confirmation-required")
