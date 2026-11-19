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
    const result = parseCaptionConfig({ chunk: { maxSeconds: 10 } });
    expect(result.ok && result.config.chunk).toEqual({ minSeconds: 2, maxSeconds: 10, silenceSeconds: 0.3, interimSeconds: 1 });
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
    const result = parseCaptionConfig({ chunk: { minSeconds: 8, maxSeconds: 4 } });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.errors.join("\n")).toContain("chunk");
  });

  it("keeps dragged overlay offsets within one anchor size", () => {
    expect(parseCaptionConfig({ overlay: { offsetX: -0.4, offsetY: 0.2 } }).ok).toBe(true);
    expect(parseCaptionConfig({ overlay: { offsetX: 1.5 } }).ok).toBe(false);
  });

  it("rejects unknown top-level keys", () => {
    expect(parseCaptionConfig({ unexpected: true }).ok).toBe(false);
  });
});
