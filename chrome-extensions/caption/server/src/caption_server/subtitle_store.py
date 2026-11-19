import json
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .atomic_file import write_atomically
from .overwrite_policy import SubtitleStatus, decide_overwrite
from .schemas import URL_KEY_PATTERN, SubtitleDocument, SubtitleMeta, SubtitleMetaInput
from .srt_codec import Cue, parse_srt, serialize_srt
from .subtitle_history import SubtitleHistory
from .subtitle_merge import Interval, Placement, Recording, combine_recordings, covered_seconds

_URL_KEY = re.compile(URL_KEY_PATTERN)
_INITIAL_REVISION = 0


class InvalidUrlKeyError(ValueError):
    pass


class RevisionConflictError(Exception):
    def __init__(self, current_revision: int) -> None:
        super().__init__(f"Revision conflict, current revision is {current_revision}")
        self.current_revision = current_revision


@dataclass(frozen=True)
class SubtitleWrite:
    meta: SubtitleMetaInput
    cues: list[Cue]
    placement: Placement
    expected_revision: int | None
    confirm_overwrite: bool
    engine: dict[str, str]


@dataclass(frozen=True)
class WriteOutcome:
    written: bool
    revision: int
    reason: str | None = None


@dataclass(frozen=True)
class UndoOutcome:
    restored: bool
    revision: int
    remaining_versions: int


@dataclass(frozen=True)
class StorePolicy:
    history_limit: int = 5
    completion_ratio: float = 0.95


@dataclass(frozen=True)
class _MergeBase:
    recording: Recording | None
    version: str


def _recording_of(document: SubtitleDocument | None) -> Recording | None:
    if document is None:
        return None
    return Recording(parse_srt(document.srt), list(document.meta.covered_intervals))


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FileSubtitleStore:
    def __init__(self, root: Path, audit_log: Path, policy: StorePolicy = StorePolicy()) -> None:
        self._root = root
        self._audit_log = audit_log
        self._policy = policy
        self._history = SubtitleHistory(root, policy.history_limit)
        self._root.mkdir(parents=True, exist_ok=True)
        self._key_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._registry_lock = threading.Lock()
        self._audit_lock = threading.Lock()

    def _lock_for(self, url_key: str) -> threading.Lock:
        with self._registry_lock:
            return self._key_locks[url_key]

    def _paths(self, url_key: str) -> tuple[Path, Path]:
        if not _URL_KEY.fullmatch(url_key):
            raise InvalidUrlKeyError("urlKey must be 64 lowercase hex characters")
        return self._root / f"{url_key}.srt", self._root / f"{url_key}.meta.json"

    def _read(self, url_key: str) -> SubtitleDocument | None:
        srt_path, meta_path = self._paths(url_key)
        if not meta_path.exists() or not srt_path.exists():
            return None
        meta = SubtitleMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return SubtitleDocument(meta=meta, srt=srt_path.read_text(encoding="utf-8"))

    def get(self, url_key: str) -> SubtitleDocument | None:
        with self._lock_for(url_key):
            return self._read(url_key)

    def put(self, url_key: str, write: SubtitleWrite) -> WriteOutcome:
        with self._lock_for(url_key):
            existing = self._read(url_key)
            revision = existing.meta.revision if existing else _INITIAL_REVISION
            if write.expected_revision is not None and write.expected_revision != revision:
                raise RevisionConflictError(revision)
            decision = decide_overwrite(existing.meta.progress() if existing else None, write.meta.progress(), write.confirm_overwrite)
            if not decision.write:
                return WriteOutcome(False, revision, decision.reason)
            base = self._merge_base(url_key, existing, write.meta.session_id)
            combined = combine_recordings(base.recording, Recording(write.cues, list(write.meta.covered_intervals)), write.placement)
            stored = self._stored_meta(url_key, write, combined.coverage, revision + 1, base.version)
            self._save(url_key, stored, serialize_srt(combined.cues))
            self._audit("put", url_key, existing.meta if existing else None, stored)
            return WriteOutcome(True, stored.revision)

    def undo(self, url_key: str) -> UndoOutcome:
        with self._lock_for(url_key):
            existing = self._read(url_key)
            revision = existing.meta.revision if existing else _INITIAL_REVISION
            version = self._history.pop_latest(url_key)
            remaining = self._history.count(url_key)
            if version is None:
                return UndoOutcome(False, revision, remaining)
            if version.document is None:
                self._remove_current(url_key)
                self._audit("undo", url_key, existing.meta if existing else None, None)
                return UndoOutcome(True, _INITIAL_REVISION, remaining)
            restored = version.document.meta.model_copy(update={"revision": revision + 1, "base_version": None, "updated_at": _now()})
            self._save(url_key, restored, version.document.srt)
            self._audit("undo", url_key, existing.meta if existing else None, restored)
            return UndoOutcome(True, restored.revision, remaining)

    def delete(self, url_key: str) -> bool:
        with self._lock_for(url_key):
            existing = self._read(url_key)
            self._remove_current(url_key)
            self._history.clear(url_key)
            if existing:
                self._audit("delete", url_key, existing.meta, None)
            return existing is not None

    def _merge_base(self, url_key: str, existing: SubtitleDocument | None, session_id: str) -> _MergeBase:
        if existing and existing.meta.session_id == session_id and existing.meta.base_version:
            saved = self._history.load(url_key, existing.meta.base_version)
            if saved is not None:
                return _MergeBase(_recording_of(saved.document), saved.name)
        return _MergeBase(_recording_of(existing), self._history.save(url_key, existing))

    def _status(self, requested: SubtitleStatus, coverage: list[Interval], media_duration: float | None) -> SubtitleStatus:
        if requested == "complete":
            return "complete"
        if media_duration and covered_seconds(coverage) >= media_duration * self._policy.completion_ratio:
            return "complete"
        return "partial"

    def _stored_meta(self, url_key: str, write: SubtitleWrite, coverage: list[Interval], revision: int, base_version: str) -> SubtitleMeta:
        fields = write.meta.model_dump()
        fields.update(
            status=self._status(write.meta.status, coverage, write.meta.media_duration),
            covered_intervals=coverage,
            covered_seconds=covered_seconds(coverage),
        )
        return SubtitleMeta(
            **fields,
            url_key=url_key,
            revision=revision,
            engine=write.engine if write.meta.source == "asr" else {},
            updated_at=_now(),
            base_version=base_version,
        )

    def _save(self, url_key: str, meta: SubtitleMeta, srt: str) -> None:
        srt_path, meta_path = self._paths(url_key)
        write_atomically(srt_path, srt)
        write_atomically(meta_path, meta.model_dump_json(by_alias=True, indent=1))

    def _remove_current(self, url_key: str) -> None:
        for path in self._paths(url_key):
            path.unlink(missing_ok=True)

    def _audit(self, action: str, url_key: str, before: SubtitleMeta | None, after: SubtitleMeta | None) -> None:
        def summary(meta: SubtitleMeta | None) -> dict[str, object] | None:
            if meta is None:
                return None
            return {"revision": meta.revision, "status": meta.status, "coveredSeconds": meta.covered_seconds, "sessionId": meta.session_id}

        entry = {"at": _now(), "action": action, "urlKey": url_key, "before": summary(before), "after": summary(after)}
        with self._audit_lock, self._audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
