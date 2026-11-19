import { describe, expect, it } from "vitest";
import { canonicalizeUrl, computeUrlKey } from "../../src/shared/url/url-canonicalizer";
import { DEFAULT_CANONICALIZER_OPTIONS } from "../../src/shared/url/default-site-rules";

const canonicalize = (raw: string) => canonicalizeUrl(raw, DEFAULT_CANONICALIZER_OPTIONS);
const YOUTUBE_CANONICAL = "https://www.youtube.com/watch?v=aircAruvnKk";

describe("canonicalizeUrl for YouTube", () => {
  it.each([
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://youtube.com/watch?v=aircAruvnKk&t=30s&list=PLxyz&index=2",
    "https://m.youtube.com/watch?feature=share&v=aircAruvnKk#comments",
    "https://youtu.be/aircAruvnKk?si=abc&t=10",
    "http://www.youtube.com/watch?v=aircAruvnKk",
  ])("maps %s to one key", (raw) => {
    expect(canonicalize(raw)).toBe(YOUTUBE_CANONICAL);
  });

  it("maps shorts and live paths to watch URLs", () => {
    expect(canonicalize("https://www.youtube.com/shorts/abc123")).toBe("https://www.youtube.com/watch?v=abc123");
    expect(canonicalize("https://www.youtube.com/live/xyz789?si=1")).toBe("https://www.youtube.com/watch?v=xyz789");
  });
});

describe("canonicalizeUrl for other sites", () => {
  it("strips tracking params, drops the fragment and sorts the rest", () => {
    expect(canonicalize("https://Example.com:443/video?b=2&utm_source=x&a=1&fbclid=y#t=5")).toBe("https://example.com/video?a=1&b=2");
  });

  it("keeps identifying query params on unknown sites", () => {
    expect(canonicalize("https://videos.example.org/play.php?id=42")).not.toBe(canonicalize("https://videos.example.org/play.php?id=43"));
  });

  it("keeps only the part param on bilibili", () => {
    expect(canonicalize("https://www.bilibili.com/video/BV1xx/?p=2&spm_id_from=333&vd_source=abc")).toBe("https://www.bilibili.com/video/BV1xx/?p=2");
  });

  it.each(["not a url", "chrome://settings", "file:///tmp/a.mp4", "about:blank"])("rejects %s", (raw) => {
    expect(canonicalize(raw)).toBeNull();
  });
});

describe("computeUrlKey", () => {
  it("returns a stable lowercase sha256 hex digest", async () => {
    const key = await computeUrlKey(YOUTUBE_CANONICAL);
    expect(key).toMatch(/^[0-9a-f]{64}$/);
    expect(await computeUrlKey(YOUTUBE_CANONICAL)).toBe(key);
    expect(await computeUrlKey(`${YOUTUBE_CANONICAL}x`)).not.toBe(key);
  });
});
