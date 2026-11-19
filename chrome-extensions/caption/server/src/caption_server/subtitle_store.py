import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .overwrite_policy import decide_overwrite
from .schemas import URL_KEY_PATTERN, SubtitleDocument, SubtitleMeta, SubtitleMetaInput

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
    srt: str
    expected_revision: int | None
    confirm_overwrite: bool
    engine: dict[str, str]


@dataclass(frozen=True)
class WriteOutcome:
    written: bool
    revision: int
    reason: str | None = None


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


class FileSubtitleStore:
    def __init__(self, root: Path, audit_log: Path) -> None:
        self._root = root
        self._audit_log = audit_log
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

    def put(self, url_key: str, request: SubtitleWrite) -> WriteOutcome:
        with self._lock_for(url_key):
            existing = self._read(url_key)
            current_revision = existing.meta.revision if existing else _INITIAL_REVISION
            if request.expected_revision is not None and request.expected_revision != current_revision:
                raise RevisionConflictError(current_revision)
            meta = request.meta
            decision = decide_overwrite(existing.meta.progress() if existing else None, meta.progress(), request.confirm_overwrite)
            if not decision.write:
                return WriteOutcome(False, current_revision, decision.reason)
            stored = SubtitleMeta(
                **meta.model_dump(),
                url_key=url_key,
                revision=current_revision + 1,
                engine=request.engine if meta.source == "asr" else {},
                updated_at=datetime.now(UTC).isoformat(),
            )
            self._write(url_key, stored, request.srt, existing)
            self._audit("put", url_key, existing.meta if existing else None, stored)
            return WriteOutcome(True, stored.revision)

    def delete(self, url_key: str) -> bool:
        with self._lock_for(url_key):
            existing = self._read(url_key)
            srt_path, meta_path = self._paths(url_key)
            for path in (meta_path, srt_path, srt_path.with_suffix(".srt.prev")):
                path.unlink(missing_ok=True)
            if existing:
                self._audit("delete", url_key, existing.meta, None)
            return existing is not None

    def _write(self, url_key: str, meta: SubtitleMeta, srt: str, existing: SubtitleDocument | None) -> None:
        srt_path, meta_path = self._paths(url_key)
        if existing:
            _atomic_write(srt_path.with_suffix(".srt.prev"), existing.srt)
        _atomic_write(srt_path, srt)
        _atomic_write(meta_path, meta.model_dump_json(by_alias=True, indent=1))

    def _audit(self, action: str, url_key: str, before: SubtitleMeta | None, after: SubtitleMeta | None) -> None:
        def summary(meta: SubtitleMeta | None) -> dict[str, object] | None:
            if meta is None:
                return None
            return {"revision": meta.revision, "status": meta.status, "coveredSeconds": meta.covered_seconds, "sessionId": meta.session_id}

        entry = {"at": datetime.now(UTC).isoformat(), "action": action, "urlKey": url_key, "before": summary(before), "after": summary(after)}
        with self._audit_lock, self._audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
