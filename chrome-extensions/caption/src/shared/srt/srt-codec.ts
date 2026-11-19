import { compareCuesByStart, type Cue } from "../cue/cue";

const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;
const TIMESTAMP_PATTERN = /^(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})$/;
const TIMING_LINE_PATTERN = /^(\S+)\s+-->\s+(\S+)/;
const BLOCK_SEPARATOR = /\r?\n\s*\r?\n/;
const LINE_SEPARATOR = /\r?\n/;

function pad(value: number, width: number): string {
  return String(value).padStart(width, "0");
}

export function formatSrtTimestamp(seconds: number): string {
  const totalMilliseconds = Math.max(0, Math.round(seconds * MILLISECONDS_PER_SECOND));
  const milliseconds = totalMilliseconds % MILLISECONDS_PER_SECOND;
  const totalSeconds = Math.floor(totalMilliseconds / MILLISECONDS_PER_SECOND);
  const hours = Math.floor(totalSeconds / SECONDS_PER_HOUR);
  const minutes = Math.floor((totalSeconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  const secs = totalSeconds % SECONDS_PER_MINUTE;
  return `${pad(hours, 2)}:${pad(minutes, 2)}:${pad(secs, 2)},${pad(milliseconds, 3)}`;
}

export function parseSrtTimestamp(value: string): number {
  const match = TIMESTAMP_PATTERN.exec(value.trim());
  if (!match) {
    throw new Error(`Invalid SRT timestamp: ${value}`);
  }
  const [, hours, minutes, seconds, milliseconds] = match.map(Number) as [number, number, number, number, number];
  return hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds + milliseconds / MILLISECONDS_PER_SECOND;
}

function serializeCue(cue: Cue, index: number): string {
  return `${index + 1}\n${formatSrtTimestamp(cue.start)} --> ${formatSrtTimestamp(cue.end)}\n${cue.text}\n`;
}

export function serializeSrt(cues: readonly Cue[]): string {
  return [...cues].sort(compareCuesByStart).map(serializeCue).join("\n");
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
