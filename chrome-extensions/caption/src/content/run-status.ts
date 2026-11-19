import type { TabStatus } from "../shared/messages";
import { coveredSeconds, unionIntervals, type Interval } from "../shared/progress/coverage";
import type { RecognitionSession } from "../shared/progress/recognition-session";

export interface RunStatusInput {
  readonly mode: TabStatus["mode"];
  readonly canonicalUrl: string;
  readonly urlKey: string;
  readonly savedCoverage: readonly Interval[];
  readonly savedCueCount: number;
  readonly session: RecognitionSession | null;
  readonly recording: boolean;
  readonly askingOverwrite: boolean;
  readonly media: HTMLMediaElement | null;
}

function finiteOrNull(value: number | undefined): number | null {
  return value !== undefined && Number.isFinite(value) && value >= 0 ? value : null;
}

function saveMessage(input: RunStatusInput): string | null {
  if (input.session?.savingDeclined) {
    return "Not saving this recording: the existing subtitles were kept";
  }
  if (input.askingOverwrite) {
    return "Waiting for your answer in the caption box";
  }
  return input.session?.awaitingOverwriteDecision ? "Not saved yet: press Save, or answer when you pause or stop" : null;
}

export function buildRunStatus(input: RunStatusInput): TabStatus {
  const coverage = unionIntervals([...input.savedCoverage, ...(input.session?.liveCoverage ?? [])]);
  return {
    mode: input.mode,
    canonicalUrl: input.canonicalUrl,
    urlKey: input.urlKey,
    coveredSeconds: coveredSeconds(coverage),
    cueCount: input.session ? input.session.liveCues.length : input.savedCueCount,
    message: saveMessage(input),
    recording: input.recording,
    canSave: input.session !== null && !input.session.savingDeclined && input.session.liveCues.length > 0,
    mediaDuration: finiteOrNull(input.media?.duration),
    currentTime: finiteOrNull(input.media?.currentTime),
    coverage: coverage.map(([start, end]) => [start, end]),
  };
}
