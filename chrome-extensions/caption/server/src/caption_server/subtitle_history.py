import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .atomic_file import write_atomically
from .schemas import SubtitleDocument

_VERSION_NAME = re.compile(r"^\d{8}T\d{12}Z$")
_VERSION_SUFFIX = ".json"


@dataclass(frozen=True)
class HistoryVersion:
    name: str
    document: SubtitleDocument | None


class SubtitleHistory:
    def __init__(self, root: Path, limit: int) -> None:
        self._root = root
        self._limit = limit

    def _directory(self, url_key: str) -> Path:
        return self._root / f"{url_key}.history"

    def _names(self, url_key: str) -> list[str]:
        directory = self._directory(url_key)
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob(f"*{_VERSION_SUFFIX}") if _VERSION_NAME.fullmatch(path.stem))

    def save(self, url_key: str, document: SubtitleDocument | None) -> str:
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {"document": document.model_dump(by_alias=True) if document else None}
        write_atomically(self._directory(url_key) / f"{name}{_VERSION_SUFFIX}", json.dumps(payload, ensure_ascii=False))
        for stale in self._names(url_key)[: -self._limit]:
            (self._directory(url_key) / f"{stale}{_VERSION_SUFFIX}").unlink(missing_ok=True)
        return name

    def load(self, url_key: str, name: str) -> HistoryVersion | None:
        path = self._directory(url_key) / f"{name}{_VERSION_SUFFIX}"
        if not _VERSION_NAME.fullmatch(name) or not path.exists():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))["document"]
        return HistoryVersion(name, SubtitleDocument.model_validate(document) if document else None)

    def pop_latest(self, url_key: str) -> HistoryVersion | None:
        names = self._names(url_key)
        if not names:
            return None
        version = self.load(url_key, names[-1])
        (self._directory(url_key) / f"{names[-1]}{_VERSION_SUFFIX}").unlink(missing_ok=True)
        return version

    def count(self, url_key: str) -> int:
        return len(self._names(url_key))

    def clear(self, url_key: str) -> None:
        shutil.rmtree(self._directory(url_key), ignore_errors=True)
