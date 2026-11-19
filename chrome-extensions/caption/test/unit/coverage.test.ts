import { describe, expect, it } from "vitest";
import { addInterval, coveredSeconds, isCovered, isRecognitionComplete } from "../../src/shared/progress/coverage";

describe("addInterval", () => {
  it("merges overlapping and touching intervals", () => {
    let intervals = addInterval([], [0, 6]);
    intervals = addInterval(intervals, [6, 12]);
    intervals = addInterval(intervals, [10, 14]);
    expect(intervals).toEqual([[0, 14]]);
  });

  it("keeps disjoint intervals sorted after seeking backwards", () => {
    const intervals = addInterval(addInterval([], [60, 70]), [0, 5]);
    expect(intervals).toEqual([
      [0, 5],
      [60, 70],
    ]);
  });

  it("bridges small recorder gaps within tolerance", () => {
    expect(addInterval([[0, 5.97]], [6, 12], 0.05)).toEqual([[0, 12]]);
    expect(addInterval([[0, 5.9]], [6, 12], 0.05)).toEqual([
      [0, 5.9],
      [6, 12],
    ]);
  });

  it("ignores empty or inverted intervals", () => {
    expect(addInterval([[0, 1]], [5, 5])).toEqual([[0, 1]]);
    expect(addInterval([[0, 1]], [5, 3])).toEqual([[0, 1]]);
  });

  it("does not count replayed ranges twice", () => {
    const once = addInterval([], [0, 30]);
    expect(coveredSeconds(addInterval(once, [10, 20]))).toBe(30);
  });
});

describe("isCovered", () => {
  it("treats interval end as exclusive", () => {
    expect(isCovered([[0, 5]], 0)).toBe(true);
    expect(isCovered([[0, 5]], 4.99)).toBe(true);
    expect(isCovered([[0, 5]], 5)).toBe(false);
  });
});

describe("isRecognitionComplete", () => {
  const base = { coveredSeconds: 96, mediaDuration: 100, ended: true, completionRatio: 0.95 };

  it("requires the ended event and enough coverage", () => {
    expect(isRecognitionComplete(base)).toBe(true);
    expect(isRecognitionComplete({ ...base, ended: false })).toBe(false);
    expect(isRecognitionComplete({ ...base, coveredSeconds: 90 })).toBe(false);
  });

  it("never completes live streams without a finite duration", () => {
    expect(isRecognitionComplete({ ...base, mediaDuration: null })).toBe(false);
    expect(isRecognitionComplete({ ...base, mediaDuration: Number.POSITIVE_INFINITY })).toBe(false);
  });
});
