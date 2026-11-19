import json
import threading

import pytest

from caption_server.schemas import SubtitleMetaInput
from caption_server.subtitle_store import FileSubtitleStore, InvalidUrlKeyError, RevisionConflictError, SubtitleWrite

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


def put(store: FileSubtitleStore, meta_input: SubtitleMetaInput, srt: str, expected_revision: int | None = None, confirm: bool = False):
    return store.put(KEY, SubtitleWrite(meta_input, srt, expected_revision, confirm, ENGINE))


def test_first_write_creates_srt_meta_and_audit(store: FileSubtitleStore, settings) -> None:
    outcome = put(store, meta(10), "1\n00:00:00,000 --> 00:00:02,000\nhi\n")
    assert (outcome.written, outcome.revision) == (True, 1)
    document = store.get(KEY)
    assert document is not None
    assert document.srt.endswith("hi\n")
    assert document.meta.engine == ENGINE
    assert (settings.subtitles_dir / f"{KEY}.srt").exists()
    entry = json.loads(settings.audit_log.read_text().splitlines()[0])
    assert entry["action"] == "put" and entry["after"]["revision"] == 1


def test_other_session_needs_confirmation_to_overwrite(store: FileSubtitleStore) -> None:
    put(store, meta(30, "old", 4), "old")
    refused = put(store, meta(300, "new", 1), "new")
    assert (refused.written, refused.reason) == (False, "confirmation-required")
    assert store.get(KEY).srt == "old"
    confirmed = put(store, meta(10, "new", 2), "new", confirm=True)
    assert (confirmed.written, confirmed.revision) == (True, 2)
    assert store.get(KEY).srt == "new"
    assert put(store, meta(12, "new", 3), "newer").written is True


def test_previous_version_is_kept_as_backup(store: FileSubtitleStore, settings) -> None:
    put(store, meta(10, seq=1), "first")
    put(store, meta(20, seq=2), "second")
    assert (settings.subtitles_dir / f"{KEY}.srt.prev").read_text() == "first"


def test_expected_revision_mismatch_raises_conflict(store: FileSubtitleStore) -> None:
    put(store, meta(10, seq=1), "first")
    with pytest.raises(RevisionConflictError) as conflict:
        put(store, meta(20, seq=2), "second", 0)
    assert conflict.value.current_revision == 1
    assert put(store, meta(20, seq=2), "second", 1).revision == 2


def test_complete_record_is_replaced_only_after_confirmation(store: FileSubtitleStore) -> None:
    put(store, meta(600, status="complete"), "done")
    assert put(store, meta(5, "other", 1), "again").reason == "confirmation-required"
    assert put(store, meta(5, "other", 1), "again", confirm=True).written is True


@pytest.mark.parametrize("bad_key", ["../etc/passwd", "A" * 64, "a" * 63, "a" * 64 + ".srt", ""])
def test_rejects_keys_that_are_not_sha256_hex(store: FileSubtitleStore, bad_key: str) -> None:
    with pytest.raises(InvalidUrlKeyError):
        store.get(bad_key)


def test_delete_removes_all_files(store: FileSubtitleStore, settings) -> None:
    put(store, meta(10, seq=1), "first")
    put(store, meta(20, seq=2), "second")
    assert store.delete(KEY) is True
    assert list(settings.subtitles_dir.iterdir()) == []
    assert store.delete(KEY) is False


def test_concurrent_writers_never_overwrite_each_other_without_confirmation(store: FileSubtitleStore) -> None:
    def writer(session: str) -> None:
        for seq in range(1, 60):
            put(store, meta(seq, session, seq), f"{session}:{seq}")

    threads = [threading.Thread(target=writer, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    document = store.get(KEY)
    assert document.meta.last_seq == 59
    assert document.srt == f"{document.meta.session_id}:59"
