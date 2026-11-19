import { describe, expect, it } from "vitest";
import { parseSrt, parseSrtTimestamp } from "../../src/shared/srt/srt-codec";

describe("parseSrtTimestamp", () => {
  it("accepts comma and dot millisecond separators", () => {
    expect(parseSrtTimestamp("01:02:03,042")).toBeCloseTo(3723.042, 6);
    expect(parseSrtTimestamp("01:02:03.042")).toBeCloseTo(3723.042, 6);
  });

  it("rejects malformed timestamps", () => {
    expect(() => parseSrtTimestamp("1:2:3")).toThrow("Invalid SRT timestamp");
  });
});

describe("parseSrt", () => {
  it("parses CRLF content with BOM and skips blocks without timing or text", () => {
    const content = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\nbroken block\r\n\r\n3\r\n00:00:03,000 --> 00:00:04,000\r\n\r\n";
    expect(parseSrt(content)).toEqual([{ start: 1, end: 2, text: "Hello" }]);
  });

  it("returns an empty list for empty content", () => {
    expect(parseSrt("")).toEqual([]);
  });
});
