import type { Cue } from "./cue";
import { isCovered, type Interval } from "../progress/coverage";

function lastCueStartingAtOrBefore(sortedCues: readonly Cue[], time: number): Cue | null {
  let low = 0;
  let high = sortedCues.length - 1;
  let candidate = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (sortedCues[middle]!.start <= time) {
      candidate = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return candidate < 0 ? null : sortedCues[candidate]!;
}

export function findCueAt(sortedCues: readonly Cue[], time: number): Cue | null {
  const cue = lastCueStartingAtOrBefore(sortedCues, time);
  return cue && time < cue.end ? cue : null;
}

export function findRecentCue(sortedCues: readonly Cue[], time: number, holdSeconds: number): Cue | null {
  const cue = lastCueStartingAtOrBefore(sortedCues, time);
  return cue && time - cue.end <= holdSeconds ? cue : null;
}

export interface DisplaySources {
  readonly liveCues: readonly Cue[];
  readonly liveCoverage: readonly Interval[];
  readonly fallbackCues: readonly Cue[];
  readonly liveHoldSeconds: number;
}

export function selectDisplayCue(sources: DisplaySources, time: number): Cue | null {
  if (isCovered(sources.liveCoverage, time)) {
    return findCueAt(sources.liveCues, time);
  }
  return findRecentCue(sources.liveCues, time, sources.liveHoldSeconds) ?? findCueAt(sources.fallbackCues, time);
}
