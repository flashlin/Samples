import { describe, expect, it } from "vitest";
import { formatSrtTimestamp, parseSrt, parseSrtTimestamp, serializeSrt } from "../../src/shared/srt/srt-codec";

describe("formatSrtTimestamp", () => {
  it.each([
    [0, "00:00:00,000"],
    [1.5, "00:00:01,500"],
    [59.9995, "00:01:00,000"],
    [3723.042, "01:02:03,042"],
    [36000, "10:00:00,000"],
    [-3, "00:00:00,000"],
  ])("formats %s as %s", (seconds, expected) => {
    expect(formatSrtTimestamp(seconds)).toBe(expected);
  });
});

describe("parseSrtTimestamp", () => {
  it("accepts comma and dot millisecond separators", () => {
    expect(parseSrtTimestamp("01:02:03,042")).toBeCloseTo(3723.042, 6);
    expect(parseSrtTimestamp("01:02:03.042")).toBeCloseTo(3723.042, 6);
  });

  it("rejects malformed timestamps", () => {
    expect(() => parseSrtTimestamp("1:2:3")).toThrow("Invalid SRT timestamp");
  });
});

describe("serializeSrt and parseSrt", () => {
  const cues = [
    { start: 4.2, end: 6, text: "Second cue" },
    { start: 0, end: 2.5, text: "Today we talk about transformers.\n今天我們來談 transformer。" },
    { start: 3725.1, end: 3727.25, text: "超過一小時的字幕" },
  ];

  it("sorts cues by start time and numbers them from one", () => {
    const srt = serializeSrt(cues);
    expect(srt.startsWith("1\n00:00:00,000 --> 00:00:02,500\nToday we talk")).toBe(true);
    expect(srt).toContain("3\n01:02:05,100 --> 01:02:07,250\n超過一小時的字幕");
  });

  it("round-trips multi-line bilingual cues", () => {
    expect(parseSrt(serializeSrt(cues))).toEqual([...cues].sort((a, b) => a.start - b.start));
  });

  it("parses CRLF content with BOM and skips blocks without timing or text", () => {
    const content = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\nbroken block\r\n\r\n3\r\n00:00:03,000 --> 00:00:04,000\r\n\r\n";
    expect(parseSrt(content)).toEqual([{ start: 1, end: 2, text: "Hello" }]);
  });

  it("returns an empty list for empty content", () => {
    expect(parseSrt("")).toEqual([]);
  });
});
