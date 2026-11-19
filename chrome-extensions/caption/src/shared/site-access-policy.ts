export interface SiteAccessConfig {
  readonly mode: "all-except-denylist" | "allowlist";
  readonly allowlist: readonly string[];
  readonly denylist: readonly string[];
}

const WILDCARD_PREFIX = "*.";

export function hostMatches(host: string, pattern: string): boolean {
  const normalizedHost = host.toLowerCase();
  const normalizedPattern = pattern.trim().toLowerCase();
  if (normalizedPattern.startsWith(WILDCARD_PREFIX)) {
    const base = normalizedPattern.slice(WILDCARD_PREFIX.length);
    return normalizedHost === base || normalizedHost.endsWith(`.${base}`);
  }
  return normalizedHost === normalizedPattern;
}

export function isSiteAllowed(pageUrl: string, config: SiteAccessConfig): boolean {
  if (!URL.canParse(pageUrl)) {
    return false;
  }
  const host = new URL(pageUrl).hostname;
  if (config.denylist.some((pattern) => hostMatches(host, pattern))) {
    return false;
  }
  return config.mode === "all-except-denylist" || config.allowlist.some((pattern) => hostMatches(host, pattern));
}
