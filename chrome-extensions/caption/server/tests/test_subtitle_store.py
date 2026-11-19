import json
import threading

import pytest

from caption_server.schemas import SubtitleMetaInput
from caption_server.subtitle_store import FileSubtitleStore, InvalidUrlKeyError, RevisionConflictError

KEY = "a" * 64
ENGINE = {"asr": "fake-asr", "translate": "fake-mt"}


def meta(covered: float, session: str = "s1", seq: int = 1, status: str = "partial") -> SubtitleMetaInput:
    return SubtitleMetaInput(
        canonical_url="https://www.youtube.com/watch?v=x",
        status=status,
        covered_seconds=covered,
        covered_intervals=[(0, covered)],
        media_duration=600,
        session_id=session,
        last_seq=seq,
    )


def test_first_write_creates_srt_meta_and_audit(store: FileSubtitleStore, settings) -> None:
    outcome = store.put(KEY, meta(10), "1\n00:00:00,000 --> 00:00:02,000\nhi\n", None, ENGINE)
    assert (outcome.written, outcome.revision) == (True, 1)
    document = store.get(KEY)
    assert document is not None
    assert document.srt.endswith("hi\n")
    assert document.meta.engine == ENGINE
    assert (settings.subtitles_dir / f"{KEY}.srt").exists()
    entry = json.loads(settings.audit_log.read_text().splitlines()[0])
    assert entry["action"] == "put" and entry["after"]["revision"] == 1


def test_overwrite_policy_is_enforced_server_side(store: FileSubtitleStore) -> None:
    store.put(KEY, meta(300, "old", 40), "old", None, ENGINE)
    behind = store.put(KEY, meta(120, "new", 5), "new", None, ENGINE)
    assert (behind.written, behind.reason) == (False, "not-ahead")
    assert store.get(KEY).srt == "old"
    ahead = store.put(KEY, meta(301, "new", 6), "new", None, ENGINE)
    assert (ahead.written, ahead.revision) == (True, 2)
    assert store.get(KEY).srt == "new"


def test_previous_version_is_kept_as_backup(store: FileSubtitleStore, settings) -> None:
    store.put(KEY, meta(10, seq=1), "first", None, ENGINE)
    store.put(KEY, meta(20, seq=2), "second", None, ENGINE)
    assert (settings.subtitles_dir / f"{KEY}.srt.prev").read_text() == "first"


def test_expected_revision_mismatch_raises_conflict(store: FileSubtitleStore) -> None:
    store.put(KEY, meta(10, seq=1), "first", None, ENGINE)
    with pytest.raises(RevisionConflictError) as conflict:
        store.put(KEY, meta(20, seq=2), "second", 0, ENGINE)
    assert conflict.value.current_revision == 1
    assert store.put(KEY, meta(20, seq=2), "second", 1, ENGINE).revision == 2


def test_complete_record_is_never_replaced(store: FileSubtitleStore) -> None:
    store.put(KEY, meta(600, status="complete"), "done", None, ENGINE)
    outcome = store.put(KEY, meta(600, "other", 1, "complete"), "again", None, ENGINE)
    assert (outcome.written, outcome.reason) == (False, "existing-complete")


@pytest.mark.parametrize("bad_key", ["../etc/passwd", "A" * 64, "a" * 63, "a" * 64 + ".srt", ""])
def test_rejects_keys_that_are_not_sha256_hex(store: FileSubtitleStore, bad_key: str) -> None:
    with pytest.raises(InvalidUrlKeyError):
        store.get(bad_key)


def test_delete_removes_all_files(store: FileSubtitleStore, settings) -> None:
    store.put(KEY, meta(10, seq=1), "first", None, ENGINE)
    store.put(KEY, meta(20, seq=2), "second", None, ENGINE)
    assert store.delete(KEY) is True
    assert list(settings.subtitles_dir.iterdir()) == []
    assert store.delete(KEY) is False


def test_concurrent_writers_from_two_sessions_keep_the_furthest(store: FileSubtitleStore) -> None:
    def writer(session: str, covered_values: range) -> None:
        for seq, covered in enumerate(covered_values, start=1):
            store.put(KEY, meta(covered, session, seq), f"{session}:{covered}", None, ENGINE)

    threads = [threading.Thread(target=writer, args=("a", range(1, 200, 2))), threading.Thread(target=writer, args=("b", range(2, 150, 2)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    document = store.get(KEY)
    assert document.meta.covered_seconds >= 148
    assert document.srt == f"{document.meta.session_id}:{int(document.meta.covered_seconds)}"
