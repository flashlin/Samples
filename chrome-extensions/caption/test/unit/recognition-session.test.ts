import { describe, expect, it } from "vitest";
import { RecognitionSession } from "../../src/shared/progress/recognition-session";
import { parseSrt } from "../../src/shared/srt/srt-codec";
import type { SubtitleMetaPayload } from "../../src/shared/messages";

const identity = { sessionId: "s-new", canonicalUrl: "https://www.youtube.com/watch?v=x", source: "asr" as const, completionRatio: 0.95 };

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
    expect(parseSrt(snapshot!.srt).map((cue) => cue.text)).toEqual(["第一句", "Hello world\n哈囉世界"]);
  });

  it("does not save again until something changes", () => {
    const session = new RecognitionSession(identity, null);
    session.recordSegment(0, 3, "a");
    session.acknowledgeSaved(session.takeSnapshotIfWorthSaving()!);
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
    session.recordSegment(3, 6, "b");
    expect(session.takeSnapshotIfWorthSaving()?.meta.lastSeq).toBe(2);
  });

  it("holds back until the new session covers more than the old partial", () => {
    const session = new RecognitionSession(identity, existingPartial(10));
    session.recordSegment(0, 6, "a");
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
    session.recordSegment(6, 10, "b");
    expect(session.takeSnapshotIfWorthSaving()).toBeNull();
    session.recordSegment(10, 10.5, "c");
    expect(session.takeSnapshotIfWorthSaving()?.meta.coveredSeconds).toBe(10.5);
  });

  it("keeps writing its own progress after overtaking the old partial", () => {
    const session = new RecognitionSession(identity, existingPartial(4));
    session.recordSegment(0, 5, "a");
    session.acknowledgeSaved(session.takeSnapshotIfWorthSaving()!);
    session.recordSegment(5, 6, "b");
    expect(session.takeSnapshotIfWorthSaving()?.meta.lastSeq).toBe(2);
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
