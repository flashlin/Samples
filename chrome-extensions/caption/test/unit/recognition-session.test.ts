import { describe, expect, it } from "vitest";
import { RecognitionSession } from "../../src/shared/progress/recognition-session";
import type { SubtitleMetaPayload } from "../../src/shared/messages";

const identity = { sessionId: "s-new", canonicalUrl: "https://www.youtube.com/watch?v=x", source: "asr" as const, completionRatio: 0.95, placement: "timeline" as const };

function existingPartial(coveredSeconds: number): SubtitleMetaPayload {
  return {
    canonicalUrl: identity.canonicalUrl,
    status: "partial",
    coveredSeconds,
    coveredIntervals: [[0, coveredSeconds]],
    mediaDuration: 100,
    source: "asr",
    language: null,
    sessionId: "s-old",
    lastSeq: 7,
  };
}

describe("RecognitionSession", () => {
  it("returns nothing to save before any segment", () => {
    expect(new RecognitionSession(identity, null).takeSnapshotIfWorthSaving()).toBeNull();
  });

  it("builds an SRT and partial meta from recorded segments", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 3, "第一句");
    session.recordSegment(3, 5, "");
    session.recordSegment(5, 8, "Hello world\n哈囉世界");
    session.updateMedia(100, false);
    const snapshot = session.takeSnapshotIfWorthSaving();
    expect(snapshot?.meta).toMatchObject({ status: "partial", coveredSeconds: 8, coveredIntervals: [[0, 8]], mediaDuration: 100, lastSeq: 1, sessionId: "s-new" });
    expect(snapshot!.cues.map((cue) => cue.text)).toEqual(["第一句", "Hello world\n哈囉世界"]);
  });

  it("does not save again until something changes", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 3, "a");
    session.acknowledgeSaved(session.takeSnapshotIfWorthSaving()!);
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
    session.recordSegment(3, 6, "b");
    expect(session.takeSnapshotIfWorthSaving()?.meta.lastSeq).toBe(2);
  });

  it("asks before replacing subtitles saved by another session, no matter how much it covers", () => {
    const session = new RecognitionSession(identity, existingPartial(10));
    session.recordSegment(0, 3, "");
    expect(session.awaitingOverwriteDecision).toBe(false);
    session.recordSegment(3, 60, "a");
    expect(session.awaitingOverwriteDecision).toBe(true);
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
  });

  it("saves with an explicit confirmation flag after the user agrees", () => {
    const session = new RecognitionSession(identity, existingPartial(100));
    session.recordSegment(0, 5, "a");
    session.decideOverwrite(true);
    expect(session.awaitingOverwriteDecision).toBe(false);
    const snapshot = session.takeSnapshotIfWorthSaving()!;
    expect(snapshot.confirmOverwrite).toBe(true);
    session.acknowledgeSaved(snapshot);
    session.recordSegment(5, 6, "b");
    expect(session.takeSnapshotIfWorthSaving()?.meta.lastSeq).toBe(2);
  });

  it("never saves after the user keeps the existing subtitles", () => {
    const session = new RecognitionSession(identity, existingPartial(4));
    session.recordSegment(0, 50, "a");
    session.decideOverwrite(false);
    expect(session.savingDeclined).toBe(true);
    expect(session.awaitingOverwriteDecision).toBe(false);
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
  });

  it("asks again when another session replaced the file after we saved", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 5, "a");
    session.acknowledgeSaved(session.takeSnapshotIfWorthSaving()!);
    session.markExistingReplacedElsewhere();
    expect(session.awaitingOverwriteDecision).toBe(true);
  });

  it("saves without asking when nothing was saved before", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 5, "a");
    expect(session.awaitingOverwriteDecision).toBe(false);
    expect(session.takeSnapshotIfWorthSaving()?.confirmOverwrite).toBe(false);
  });

  it("marks the subtitle complete when the media ends with enough coverage", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 96, "all");
    session.updateMedia(100, true);
    expect(session.isComplete).toBe(true);
    expect(session.takeSnapshotIfWorthSaving()?.meta.status).toBe("complete");
  });

  it("does not count a replayed range twice", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 10, "a");
    session.recordSegment(2, 8, "replayed");
    expect(session.coveredSeconds).toBe(10);
    expect(session.liveCues.map((cue) => cue.text)).toEqual(["replayed"]);
  });

  it("ignores live stream durations", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 10, "a");
    session.updateMedia(Number.POSITIVE_INFINITY, true);
    expect(session.isComplete).toBe(false);
    expect(session.takeSnapshotIfWorthSaving()?.meta.mediaDuration).toBeNull();
  });
});

describe("RecognitionSession source switching", () => {
  it("keeps one session id and coverage when the source changes on the same page", () => {
    const session = new RecognitionSession({ ...identity, source: "youtube-captions" }, null);
    session.recordSegment(0, 5, "caption");
    const first = session.takeSnapshotIfWorthSaving()!;
    session.acknowledgeSaved(first);
    session.useSource("asr");
    session.recordSegment(5, 9, "speech");
    const second = session.takeSnapshotIfWorthSaving()!;
    expect(second.meta.sessionId).toBe(first.meta.sessionId);
    expect(second.meta.source).toBe("asr");
    expect(second.meta.coveredSeconds).toBe(9);
    expect(second.meta.lastSeq).toBe(first.meta.lastSeq + 1);
  });

  it("marks the session dirty when only the source changes", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 5, "a");
    session.acknowledgeSaved(session.takeSnapshotIfWorthSaving()!);
    session.useSource("asr");
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
    session.useSource("youtube-captions");
    expect(session.takeSnapshotIfWorthSaving()?.meta.source).toBe("youtube-captions");
  });
});
