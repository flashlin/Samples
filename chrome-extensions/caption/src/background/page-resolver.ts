import type { CaptionConfig } from "../shared/config";
import { subtitleMetaSchema, type ResolvedPage } from "../shared/messages";
import { isSiteAllowed } from "../shared/site-access-policy";
import { parseSrt } from "../shared/srt/srt-codec";
import { canonicalizeUrl, computeUrlKey } from "../shared/url/url-canonicalizer";
import type { CaptionServerClient } from "./caption-server-client";
import { toContentSettings } from "./config-repository";

export interface PageIdentity {
  readonly canonicalUrl: string;
  readonly urlKey: string;
}

export async function identifyPage(pageUrl: string, config: CaptionConfig): Promise<PageIdentity | null> {
  const canonicalUrl = canonicalizeUrl(pageUrl, config.url);
  if (canonicalUrl === null) {
    return null;
  }
  return { canonicalUrl, urlKey: await computeUrlKey(canonicalUrl) };
}

export async function resolvePage(pageUrl: string, config: CaptionConfig, client: CaptionServerClient): Promise<ResolvedPage> {
  if (!isSiteAllowed(pageUrl, config.sites)) {
    return { mode: "disabled", reason: "Site is excluded by the caption site policy" };
  }
  const identity = await identifyPage(pageUrl, config);
  if (identity === null) {
    return { mode: "disabled", reason: "Page URL is not an http(s) URL" };
  }
  const stored = await client.getSubtitle(identity.urlKey);
  return {
    mode: stored?.meta.status === "complete" ? "playback" : "recognize",
    urlKey: identity.urlKey,
    canonicalUrl: identity.canonicalUrl,
    cues: stored ? parseSrt(stored.srt) : [],
    revision: stored?.meta.revision ?? 0,
    existing: stored ? subtitleMetaSchema.parse(stored.meta) : null,
    settings: toContentSettings(config),
  };
}
