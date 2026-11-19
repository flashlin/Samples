export interface PathAlias {
  readonly pattern: string;
  readonly path: string;
  readonly param: string;
}

export interface SiteRule {
  readonly hosts: readonly string[];
  readonly canonicalOrigin: string;
  readonly keepParams: readonly string[];
  readonly pathAliases: readonly PathAlias[];
}

export interface CanonicalizerOptions {
  readonly stripParams: readonly string[];
  readonly siteRules: readonly SiteRule[];
}

const SUPPORTED_PROTOCOLS = new Set(["http:", "https:"]);
const WILDCARD_SUFFIX = "*";

function matchesParamName(name: string, pattern: string): boolean {
  return pattern.endsWith(WILDCARD_SUFFIX) ? name.startsWith(pattern.slice(0, -1)) : name === pattern;
}

function sortedParams(entries: Iterable<[string, string]>): URLSearchParams {
  return new URLSearchParams([...entries].sort(([a, av], [b, bv]) => a.localeCompare(b) || av.localeCompare(bv)));
}

function applyPathAliases(url: URL, aliases: readonly PathAlias[]): void {
  for (const alias of aliases) {
    const match = new RegExp(alias.pattern).exec(url.pathname);
    if (match?.[1]) {
      url.searchParams.set(alias.param, decodeURIComponent(match[1]));
      url.pathname = alias.path;
      return;
    }
  }
}

function canonicalizeWithRule(url: URL, rule: SiteRule): string {
  applyPathAliases(url, rule.pathAliases);
  const kept = [...url.searchParams].filter(([name]) => rule.keepParams.includes(name));
  const query = sortedParams(kept).toString();
  return `${rule.canonicalOrigin}${url.pathname}${query ? `?${query}` : ""}`;
}

function canonicalizeGeneric(url: URL, stripParams: readonly string[]): string {
  const kept = [...url.searchParams].filter(([name]) => !stripParams.some((pattern) => matchesParamName(name, pattern)));
  const query = sortedParams(kept).toString();
  return `${url.origin}${url.pathname}${query ? `?${query}` : ""}`;
}

export function canonicalizeUrl(raw: string, options: CanonicalizerOptions): string | null {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (!SUPPORTED_PROTOCOLS.has(url.protocol)) {
    return null;
  }
  url.hash = "";
  const rule = options.siteRules.find((candidate) => candidate.hosts.includes(url.hostname));
  return rule ? canonicalizeWithRule(url, rule) : canonicalizeGeneric(url, options.stripParams);
}

export async function computeUrlKey(canonicalUrl: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonicalUrl));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
