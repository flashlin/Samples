import { describe, expect, it } from "vitest";
import { findCueAt, findRecentCue, selectDisplayCue } from "../../src/shared/cue/cue-timeline";
import { mergeCues } from "../../src/shared/cue/cue-merger";

const cues = [
  { start: 0, end: 2, text: "a" },
  { start: 2, end: 4, text: "b" },
  { start: 10, end: 12, text: "c" },
];

describe("findCueAt", () => {
  it.each([
    [0, "a"],
    [1.99, "a"],
    [2, "b"],
    [11, "c"],
  ])("finds the cue active at %s", (time, text) => {
    expect(findCueAt(cues, time)?.text).toBe(text);
  });

  it.each([-1, 4, 7, 12, 100])("returns null in gaps at %s", (time) => {
    expect(findCueAt(cues, time)).toBeNull();
  });

  it("handles an empty list", () => {
    expect(findCueAt([], 1)).toBeNull();
  });
});

describe("findRecentCue", () => {
  it("keeps showing the latest cue for the hold window after it ends", () => {
    expect(findRecentCue(cues, 5, 2)?.text).toBe("b");
    expect(findRecentCue(cues, 6.5, 2)).toBeNull();
    expect(findRecentCue(cues, 11, 2)?.text).toBe("c");
  });
});

describe("selectDisplayCue", () => {
  const fallbackCues = [{ start: 0, end: 30, text: "old partial" }];
  const liveCues = [{ start: 0, end: 5, text: "live" }];
  const sources = { liveCues, liveCoverage: [[0, 8]] as [number, number][], fallbackCues, liveHoldSeconds: 3 };

  it("prefers live cues inside live coverage", () => {
    expect(selectDisplayCue(sources, 3)?.text).toBe("live");
  });

  it("shows nothing for covered silence instead of stale fallback text", () => {
    expect(selectDisplayCue(sources, 6)).toBeNull();
  });

  it("keeps the freshly recognized cue while recognition lags behind playback", () => {
    expect(selectDisplayCue({ ...sources, liveCoverage: [[0, 5]] }, 7)?.text).toBe("live");
  });

  it("falls back to the old partial outside live coverage once the hold expires", () => {
    expect(selectDisplayCue(sources, 20)?.text).toBe("old partial");
  });
});

describe("mergeCues", () => {
  it("replaces cues overlapping a re-recognized range and keeps order", () => {
    const merged = mergeCues(cues, [{ start: 1, end: 3, text: "new" }]);
    expect(merged.map((cue) => cue.text)).toEqual(["new", "c"]);
  });

  it("keeps touching cues that do not overlap", () => {
    const merged = mergeCues(cues, [{ start: 4, end: 10, text: "gap" }]);
    expect(merged.map((cue) => cue.text)).toEqual(["a", "b", "gap", "c"]);
  });
});
