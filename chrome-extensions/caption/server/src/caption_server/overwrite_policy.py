from dataclasses import dataclass
from typing import Literal

SubtitleStatus = Literal["partial", "complete"]
OverwriteRejection = Literal["empty-candidate", "existing-complete", "stale-sequence", "not-ahead"]


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


def decide_overwrite(existing: SubtitleProgress | None, candidate: SubtitleProgress) -> OverwriteDecision:
    if candidate.status == "partial" and candidate.covered_seconds <= 0:
        return OverwriteDecision(False, "empty-candidate")
    if existing is None:
        return _ACCEPT
    if existing.status == "complete":
        return OverwriteDecision(False, "existing-complete")
    same_session = existing.session_id == candidate.session_id
    if same_session and candidate.last_seq <= existing.last_seq:
        return OverwriteDecision(False, "stale-sequence")
    if same_session or candidate.status == "complete":
        return _ACCEPT
    return _ACCEPT if candidate.covered_seconds > existing.covered_seconds else OverwriteDecision(False, "not-ahead")
