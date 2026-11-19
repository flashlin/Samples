import { z } from "zod";
import { DEFAULT_CANONICALIZER_OPTIONS } from "./url/default-site-rules";
import type { SiteRule } from "./url/url-canonicalizer";

export const CONFIG_STORAGE_KEY = "captionConfig";
export const LOOPBACK_HOSTS: ReadonlySet<string> = new Set(["127.0.0.1", "localhost", "[::1]"]);

function isSecureOrLoopback(value: string): boolean {
  if (!URL.canParse(value)) {
    return false;
  }
  const url = new URL(value);
  return url.protocol === "https:" || (url.protocol === "http:" && LOOPBACK_HOSTS.has(url.hostname));
}

function cloneSiteRule(rule: SiteRule) {
  return {
    hosts: [...rule.hosts],
    canonicalOrigin: rule.canonicalOrigin,
    keepParams: [...rule.keepParams],
    pathAliases: rule.pathAliases.map((alias) => ({ ...alias })),
  };
}

const endpointSchema = z.url().refine(isSecureOrLoopback, { message: "Endpoint must use https unless it is a loopback address" });

const pathAliasSchema = z.object({ pattern: z.string().min(1), path: z.string().startsWith("/"), param: z.string().min(1) });

const siteRuleSchema = z.object({
  hosts: z.array(z.string().min(1)).min(1),
  canonicalOrigin: z.url(),
  keepParams: z.array(z.string()),
  pathAliases: z.array(pathAliasSchema),
});

export const captionConfigSchema = z
  .object({
    server: z
      .object({
        endpoint: endpointSchema.default("http://127.0.0.1:8765"),
        token: z.string().default(""),
        requestTimeoutSeconds: z.number().positive().default(30),
      })
      .prefault({}),
    flush: z.object({ intervalSeconds: z.number().positive().default(10) }).prefault({}),
    completion: z.object({ ratio: z.number().gt(0).max(1).default(0.95) }).prefault({}),
    sites: z
      .object({
        mode: z.enum(["all-except-denylist", "allowlist"]).default("all-except-denylist"),
        allowlist: z.array(z.string()).default([]),
        denylist: z.array(z.string()).default([]),
      })
      .prefault({}),
    url: z
      .object({
        stripParams: z.array(z.string()).default(() => [...DEFAULT_CANONICALIZER_OPTIONS.stripParams]),
        siteRules: z.array(siteRuleSchema).default(() => DEFAULT_CANONICALIZER_OPTIONS.siteRules.map(cloneSiteRule)),
      })
      .prefault({}),
    overlay: z
      .object({
        fontSizePx: z.number().int().min(10).max(64).default(20),
        maxSentences: z.number().int().min(1).max(4).default(1),
        opacity: z.number().min(0.2).max(1).default(0.85),
        offsetX: z.number().min(-1).max(1).default(0),
        offsetY: z.number().min(-1).max(1).default(0),
      })
      .prefault({}),
    features: z
      .object({
        autoStart: z.boolean().default(true),
        tabCaptureFallback: z.boolean().default(true),
        useYoutubeCaptions: z.boolean().default(true),
        bilingualEnglish: z.boolean().default(true),
        recognizeOnlyAtNormalRate: z.boolean().default(true),
      })
      .prefault({}),
  });

export type CaptionConfig = z.infer<typeof captionConfigSchema>;

export type ConfigParseResult = { readonly ok: true; readonly config: CaptionConfig } | { readonly ok: false; readonly errors: readonly string[] };

export function parseCaptionConfig(raw: unknown): ConfigParseResult {
  const result = captionConfigSchema.safeParse(raw ?? {});
  if (result.success) {
    return { ok: true, config: result.data };
  }
  return { ok: false, errors: result.error.issues.map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`) };
}

export function parseCaptionConfigOrDefault(raw: unknown): { readonly config: CaptionConfig; readonly errors: readonly string[] } {
  const result = parseCaptionConfig(raw);
  return result.ok ? { config: result.config, errors: [] } : { config: captionConfigSchema.parse({}), errors: result.errors };
}
