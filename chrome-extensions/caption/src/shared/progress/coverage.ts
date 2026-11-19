export type Interval = readonly [start: number, end: number];

export function addInterval(intervals: readonly Interval[], added: Interval, toleranceSeconds = 0): Interval[] {
  const [addedStart, addedEnd] = added;
  if (!(addedEnd > addedStart)) {
    return [...intervals];
  }
  const sorted = [...intervals, added].sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [start, end] of sorted) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1] + toleranceSeconds) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

export function unionIntervals(intervals: readonly Interval[]): Interval[] {
  return intervals.reduce<Interval[]>((merged, interval) => addInterval(merged, interval), []);
}

export function findGaps(coverage: readonly Interval[], duration: number, minimumSeconds: number): Interval[] {
  const gaps: Interval[] = [];
  let cursor = 0;
  for (const [start, end] of unionIntervals(coverage)) {
    if (start - cursor >= minimumSeconds) {
      gaps.push([cursor, Math.min(start, duration)]);
    }
    cursor = Math.max(cursor, end);
  }
  if (duration - cursor >= minimumSeconds) {
    gaps.push([cursor, duration]);
  }
  return gaps;
}

export function coveredSeconds(intervals: readonly Interval[]): number {
  return intervals.reduce((total, [start, end]) => total + (end - start), 0);
}

export function isCovered(intervals: readonly Interval[], time: number): boolean {
  return intervals.some(([start, end]) => time >= start && time < end);
}

export interface CompletionInput {
  readonly coveredSeconds: number;
  readonly mediaDuration: number | null;
  readonly ended: boolean;
  readonly completionRatio: number;
}

export function isRecognitionComplete(input: CompletionInput): boolean {
  if (!input.ended || input.mediaDuration === null || !Number.isFinite(input.mediaDuration) || input.mediaDuration <= 0) {
    return false;
  }
  return input.coveredSeconds / input.mediaDuration >= input.completionRatio;
}
