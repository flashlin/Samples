import { describe, expect, it } from "vitest";
import { parseCaptionConfig } from "../../src/shared/config";

describe("parseCaptionConfig", () => {
  it("fills every default from an empty object", () => {
    const result = parseCaptionConfig({});
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.config.server.endpoint).toBe("http://127.0.0.1:8765");
    expect(result.config.sites.mode).toBe("all-except-denylist");
    expect(result.config.features.useYoutubeCaptions).toBe(true);
    expect(result.config.url.siteRules.length).toBeGreaterThan(0);
  });

  it("treats undefined as an empty config", () => {
    expect(parseCaptionConfig(undefined).ok).toBe(true);
  });

  it("merges partial nested overrides with defaults", () => {
    const result = parseCaptionConfig({ overlay: { fontSizePx: 30 } });
    expect(result.ok && result.config.overlay).toEqual({ fontSizePx: 30, maxSentences: 1, opacity: 0.85, offsetX: 0, offsetY: 0 });
  });

  it.each(["http://192.168.1.5:8765", "ftp://127.0.0.1", "not-a-url"])("rejects insecure or invalid endpoint %s", (endpoint) => {
    const result = parseCaptionConfig({ server: { endpoint } });
    expect(result.ok).toBe(false);
  });

  it("accepts https remote endpoints and loopback http", () => {
    expect(parseCaptionConfig({ server: { endpoint: "https://caption.internal.example" } }).ok).toBe(true);
    expect(parseCaptionConfig({ server: { endpoint: "http://localhost:9000" } }).ok).toBe(true);
  });

  it("reports the failing path for invalid values", () => {
    const result = parseCaptionConfig({ overlay: { fontSizePx: 2 } });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.errors.join("\n")).toContain("overlay.fontSizePx");
  });

  it("keeps dragged overlay offsets within one anchor size", () => {
    expect(parseCaptionConfig({ overlay: { offsetX: -0.4, offsetY: 0.2 } }).ok).toBe(true);
    expect(parseCaptionConfig({ overlay: { offsetX: 1.5 } }).ok).toBe(false);
  });

  it("keeps the token when a stored config still has retired sections", () => {
    const result = parseCaptionConfig({ server: { token: "abc" }, chunk: { maxSeconds: 8 }, queue: { maxRetries: 3 } });
    expect(result.ok && result.config.server.token).toBe("abc");
    expect(result.ok && "chunk" in result.config).toBe(false);
  });
});
