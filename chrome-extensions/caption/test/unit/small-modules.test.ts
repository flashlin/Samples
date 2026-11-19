import { describe, expect, it } from "vitest";
import { rootMeanSquare, shouldCutChunk } from "../../src/shared/audio/chunk-cut-policy";
import { MediaTimeMapper } from "../../src/shared/audio/media-time-mapper";
import { base64ToBytes, bytesToBase64 } from "../../src/shared/base64";
import { isOffscreenCommand, isTabBroadcast } from "../../src/shared/message-guards";
import { hostMatches, isSiteAllowed } from "../../src/shared/site-access-policy";

describe("shouldCutChunk", () => {
  const limits = { minSeconds: 2, maxSeconds: 8, silenceSeconds: 0.3 };

  it.each([
    [{ elapsedSeconds: 1, trailingSilenceSeconds: 1 }, false],
    [{ elapsedSeconds: 3, trailingSilenceSeconds: 0.1 }, false],
    [{ elapsedSeconds: 3, trailingSilenceSeconds: 0.3 }, true],
    [{ elapsedSeconds: 8, trailingSilenceSeconds: 0 }, true],
  ])("decides %o -> %s", (state, expected) => {
    expect(shouldCutChunk(state, limits)).toBe(expected);
  });

  it("computes RMS", () => {
    expect(rootMeanSquare(new Float32Array([]))).toBe(0);
    expect(rootMeanSquare(new Float32Array([0.5, -0.5]))).toBeCloseTo(0.5);
  });
});

describe("MediaTimeMapper", () => {
  it("returns null before any sample", () => {
    expect(new MediaTimeMapper().toMediaTime(10)).toBeNull();
  });

  it("extrapolates from the latest sample while playing and freezes while paused", () => {
    const mapper = new MediaTimeMapper();
    mapper.record({ wallSeconds: 100, mediaSeconds: 30, playing: true, playbackRate: 1 });
    mapper.record({ wallSeconds: 101, mediaSeconds: 31, playing: false, playbackRate: 1 });
    expect(mapper.toMediaTime(100.5)).toBeCloseTo(30.5);
    expect(mapper.toMediaTime(105)).toBe(31);
  });

  it("applies playback rate and uses the earliest sample for older times", () => {
    const mapper = new MediaTimeMapper();
    mapper.record({ wallSeconds: 100, mediaSeconds: 10, playing: true, playbackRate: 2 });
    expect(mapper.toMediaTime(101)).toBe(12);
    expect(mapper.toMediaTime(90)).toBe(0);
  });

  it("drops samples beyond capacity", () => {
    const mapper = new MediaTimeMapper(2);
    for (const wall of [1, 2, 3]) {
      mapper.record({ wallSeconds: wall, mediaSeconds: wall * 10, playing: false, playbackRate: 1 });
    }
    expect(mapper.toMediaTime(1.5)).toBe(20);
  });
});

describe("base64", () => {
  it("round-trips binary data larger than one chunk", () => {
    const bytes = new Uint8Array(70_000).map((_, index) => index % 256);
    expect(base64ToBytes(bytesToBase64(bytes))).toEqual(bytes);
  });
});

describe("site access policy", () => {
  it.each([
    ["www.bank.com", "*.bank.com", true],
    ["bank.com", "*.bank.com", true],
    ["evilbank.com", "*.bank.com", false],
    ["Meet.Google.com", "meet.google.com", true],
  ])("hostMatches(%s, %s) is %s", (host, pattern, expected) => {
    expect(hostMatches(host, pattern)).toBe(expected);
  });

  it("allows everything except the denylist by default", () => {
    const config = { mode: "all-except-denylist" as const, allowlist: [], denylist: ["*.bank.com"] };
    expect(isSiteAllowed("https://www.youtube.com/watch?v=1", config)).toBe(true);
    expect(isSiteAllowed("https://online.bank.com/video", config)).toBe(false);
    expect(isSiteAllowed("not a url", config)).toBe(false);
  });

  it("requires a match in allowlist mode and still honours the denylist", () => {
    const config = { mode: "allowlist" as const, allowlist: ["*.youtube.com"], denylist: ["music.youtube.com"] };
    expect(isSiteAllowed("https://www.youtube.com/", config)).toBe(true);
    expect(isSiteAllowed("https://music.youtube.com/", config)).toBe(false);
    expect(isSiteAllowed("https://vimeo.com/1", config)).toBe(false);
  });
});

describe("message guards", () => {
  it.each([
    [{ type: "tab-capture-started" }, true],
    [{ type: "tab-capture-segment", start: 1, end: 2, display: "x" }, true],
    [{ type: "tab-capture-segment", start: "1", end: 2, display: "x" }, false],
    [{ type: "tab-capture-stopped", message: null }, true],
    [{ type: "unknown" }, false],
    [null, false],
  ])("isTabBroadcast(%o) is %s", (value, expected) => {
    expect(isTabBroadcast(value)).toBe(expected);
  });

  it.each([
    [{ target: "offscreen", type: "offscreen-start", tabId: 1, streamId: "abc", limits: { minSeconds: 2, maxSeconds: 8, silenceSeconds: 0.3 } }, true],
    [{ target: "offscreen", type: "offscreen-start", tabId: 1, streamId: "", limits: { minSeconds: 2, maxSeconds: 8, silenceSeconds: 0.3 } }, false],
    [{ target: "offscreen", type: "offscreen-stop", tabId: 1 }, true],
    [{ target: "background", type: "offscreen-stop", tabId: 1 }, false],
    [{ target: "offscreen", type: "offscreen-stop", tabId: 1.5 }, false],
  ])("isOffscreenCommand(%o) is %s", (value, expected) => {
    expect(isOffscreenCommand(value)).toBe(expected);
  });
});
