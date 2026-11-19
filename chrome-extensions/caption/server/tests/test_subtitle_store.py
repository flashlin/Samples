import json
import threading

import pytest

from caption_server.schemas import SubtitleMetaInput
from caption_server.srt_codec import Cue, parse_srt
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


def put(store: FileSubtitleStore, meta_input: SubtitleMetaInput, text: str, expected_revision: int | None = None, confirm: bool = False):
    cues = parse_srt(text) or [Cue(0.0, max(meta_input.covered_seconds, 0.001), text)]
    return store.put(KEY, SubtitleWrite(meta_input, cues, "timeline", expected_revision, confirm, ENGINE))


def saved_text(store: FileSubtitleStore) -> str:
    return "\n".join(cue.text for cue in parse_srt(store.get(KEY).srt))


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
    assert saved_text(store) == "old"
    confirmed = put(store, meta(10, "new", 2), "new", confirm=True)
    assert (confirmed.written, confirmed.revision) == (True, 2)
    assert saved_text(store) == "new\nold"
    assert put(store, meta(12, "new", 3), "newer").written is True


def test_version_before_each_recording_is_kept_for_undo(store: FileSubtitleStore) -> None:
    put(store, meta(10, "first", 1), "first")
    put(store, meta(20, "first", 2), "first again")
    put(store, meta(5, "second", 1), "second", confirm=True)
    assert store.undo(KEY).restored is True
    assert saved_text(store) == "first again"
    assert store.undo(KEY).restored is True
    assert store.get(KEY) is None
    assert store.undo(KEY).restored is False
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
    assert saved_text(store) == f"{document.meta.session_id}:59"


def timed(covered: list[tuple[float, float]], session: str, seq: int, duration: float = 600) -> SubtitleMetaInput:
    return SubtitleMetaInput(
        canonical_url="https://www.youtube.com/watch?v=x",
        status="partial",
        covered_seconds=sum(end - start for start, end in covered),
        covered_intervals=covered,
        media_duration=duration,
        session_id=session,
        last_seq=seq,
    )


def write(cues: list[Cue], covered: list[tuple[float, float]], session: str, seq: int, placement: str = "timeline", duration: float = 600) -> SubtitleWrite:
    return SubtitleWrite(timed(covered, session, seq, duration), cues, placement, None, True, ENGINE)


def cue_texts(store: FileSubtitleStore) -> list[str]:
    return [cue.text for cue in parse_srt(store.get(KEY).srt)]


def test_confirmed_recording_fills_the_gap_and_keeps_the_rest(store: FileSubtitleStore) -> None:
    store.put(KEY, write([Cue(0, 10, "A"), Cue(40, 50, "E")], [(0, 20), (40, 60)], "old", 1))
    store.put(KEY, write([Cue(20, 30, "c"), Cue(30, 40, "d")], [(18, 42)], "new", 1))
    assert cue_texts(store) == ["A", "c", "d", "E"]
    assert store.get(KEY).meta.covered_intervals == [(0, 60)]


def test_later_saves_of_one_recording_merge_onto_the_version_before_it(store: FileSubtitleStore) -> None:
    store.put(KEY, write([Cue(0, 10, "A"), Cue(20, 30, "old C")], [(0, 30)], "old", 1))
    store.put(KEY, write([Cue(20, 25, "c1")], [(20, 25)], "new", 1))
    store.put(KEY, write([Cue(20, 26, "c1 corrected"), Cue(26, 30, "c2")], [(20, 30)], "new", 2))
    assert cue_texts(store) == ["A", "c1 corrected", "c2"]


def test_complete_when_merged_coverage_reaches_the_ratio(store: FileSubtitleStore) -> None:
    store.put(KEY, write([Cue(0, 5, "A")], [(0, 20), (40, 100)], "old", 1, duration=100))
    assert store.get(KEY).meta.status == "partial"
    store.put(KEY, write([Cue(25, 30, "B")], [(18, 42)], "new", 1, duration=100))
    assert store.get(KEY).meta.status == "complete"


def test_append_places_a_recording_without_media_time_after_the_old_one(store: FileSubtitleStore) -> None:
    store.put(KEY, write([Cue(0, 60, "first meeting")], [(0, 60)], "old", 1, "append", duration=0))
    store.put(KEY, write([Cue(0, 5, "second meeting")], [(0, 30)], "new", 1, "append", duration=0))
    cues = parse_srt(store.get(KEY).srt)
    assert [cue.text for cue in cues] == ["first meeting", "second meeting"]
    assert cues[1].start == pytest.approx(62)


def test_history_keeps_only_the_latest_versions(tmp_path) -> None:
    from caption_server.subtitle_store import StorePolicy

    limited = FileSubtitleStore(tmp_path / "subs", tmp_path / "audit.jsonl", StorePolicy(history_limit=2))
    for index in range(4):
        limited.put(KEY, write([Cue(0, 1, f"v{index}")], [(0, 1)], f"s{index}", 1))
    assert limited.undo(KEY).remaining_versions == 1
    assert cue_texts(limited) == ["v2"]
