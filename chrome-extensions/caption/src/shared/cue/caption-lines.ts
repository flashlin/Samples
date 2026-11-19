import type { Cue } from "./cue";

const SENTENCE_END = /(?<=[。！？!?]|\.(?=\s|$))\s*/u;

export function splitIntoLines(text: string | null | undefined): string[] {
  if (!text) {
    return [];
  }
  return text
    .split("\n")
    .flatMap((line) => line.split(SENTENCE_END))
    .map((line) => line.trim())
    .filter(Boolean);
}

export function previousCue(sortedCues: readonly Cue[], current: Cue, maxGapSeconds: number): Cue | null {
  let candidate: Cue | null = null;
  for (const cue of sortedCues) {
    if (cue === current || cue.start >= current.start) {
      break;
    }
    candidate = cue;
  }
  return candidate && current.start - candidate.end <= maxGapSeconds ? candidate : null;
}

export function rollingCaption(previous: string | null, current: string | null, maxLines: number): string | null {
  const lines = [...splitIntoLines(previous), ...splitIntoLines(current)];
  return lines.length > 0 ? lines.slice(-maxLines).join("\n") : null;
}
