import type { Cue } from "./cue";

const SENTENCE_END = /(?<=[。！？!?]|\.(?=\s|$))\s*/u;
const HAN = /[㐀-鿿豈-﫿]/u;
const LATIN = /[A-Za-z]/;

function isBilingualPair(lines: readonly string[]): boolean {
  const [original, translation] = lines;
  return lines.length === 2 && original !== undefined && translation !== undefined && LATIN.test(original) && !HAN.test(original) && HAN.test(translation);
}

export function splitIntoSentences(text: string | null | undefined): string[] {
  const lines = (text ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (isBilingualPair(lines)) {
    return [lines.join("\n")];
  }
  return lines
    .flatMap((line) => line.split(SENTENCE_END))
    .map((sentence) => sentence.trim())
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

export function latestSentences(previous: string | null, current: string | null, maxSentences: number): string | null {
  const sentences = [...splitIntoSentences(previous), ...splitIntoSentences(current)];
  return sentences.length > 0 ? sentences.slice(-maxSentences).join("\n") : null;
}
