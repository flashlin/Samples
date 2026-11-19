import { latestSentences, previousCue } from "../shared/cue/caption-lines";
import type { Cue } from "../shared/cue/cue";
import { findCueAt, selectDisplayCue } from "../shared/cue/cue-timeline";
import type { Interval } from "../shared/progress/coverage";

export const LIVE_CAPTION_HOLD_SECONDS = 3;

export interface DisplayState {
  readonly liveCue: Cue | null;
  readonly liveCues: readonly Cue[] | null;
  readonly liveCoverage: readonly Interval[];
  readonly savedCues: readonly Cue[];
  readonly time: number | null;
  readonly maxSentences: number;
}

function currentCue(state: DisplayState): { current: Cue | null; timeline: readonly Cue[] } {
  if (state.liveCue) {
    return { current: state.liveCue, timeline: state.liveCues ?? [] };
  }
  if (state.time === null) {
    return { current: null, timeline: [] };
  }
  if (!state.liveCues) {
    return { current: findCueAt(state.savedCues, state.time), timeline: state.savedCues };
  }
  const sources = { liveCues: state.liveCues, liveCoverage: state.liveCoverage, fallbackCues: state.savedCues, liveHoldSeconds: LIVE_CAPTION_HOLD_SECONDS };
  const current = selectDisplayCue(sources, state.time);
  return { current, timeline: current && state.liveCues.includes(current) ? state.liveCues : state.savedCues };
}

export function captionTextFor(state: DisplayState): string | null {
  const { current, timeline } = currentCue(state);
  if (!current) {
    return null;
  }
  const previous = previousCue(timeline, current, LIVE_CAPTION_HOLD_SECONDS);
  return latestSentences(previous?.text ?? null, current.text, state.maxSentences);
}
