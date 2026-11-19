import { compareCuesByStart, type Cue } from "../cue/cue";

const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;
const TIMESTAMP_PATTERN = /^(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})$/;
const TIMING_LINE_PATTERN = /^(\S+)\s+-->\s+(\S+)/;
const BLOCK_SEPARATOR = /\r?\n\s*\r?\n/;
const LINE_SEPARATOR = /\r?\n/;

export function parseSrtTimestamp(value: string): number {
  const match = TIMESTAMP_PATTERN.exec(value.trim());
  if (!match) {
    throw new Error(`Invalid SRT timestamp: ${value}`);
  }
  const [, hours, minutes, seconds, milliseconds] = match.map(Number) as [number, number, number, number, number];
  return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds + milliseconds / MILLISECONDS_PER_SECOND;
}

function parseBlock(block: string): Cue | null {
  const lines = block.split(LINE_SEPARATOR);
  const timingIndex = lines.findIndex((line) => TIMING_LINE_PATTERN.test(line));
  if (timingIndex < 0) {
    return null;
  }
  const [, start, end] = TIMING_LINE_PATTERN.exec(lines[timingIndex]!) as unknown as [string, string, string];
  const text = lines.slice(timingIndex + 1).join("\n").trim();
  if (!text) {
    return null;
  }
  return { start: parseSrtTimestamp(start), end: parseSrtTimestamp(end), text };
}

export function parseSrt(content: string): Cue[] {
  return content
    .replace(/^﻿/, "")
    .trim()
    .split(BLOCK_SEPARATOR)
    .map(parseBlock)
    .filter((cue): cue is Cue => cue !== null)
    .sort(compareCuesByStart);
}
