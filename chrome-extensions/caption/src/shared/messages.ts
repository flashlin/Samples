import { z } from "zod";
import type { Cue } from "./cue/cue";
import { OVERLAY_HEIGHT_LINES } from "./overlay-layout";

const MAX_TRANSLATE_LINES = 200;

export const urlKeySchema = z.string().regex(/^[0-9a-f]{64}$/);
const intervalSchema = z.tuple([z.number().nonnegative(), z.number().nonnegative()]);

export const subtitleMetaSchema = z.object({
  canonicalUrl: z.string().min(1),
  status: z.enum(["partial", "complete"]),
  coveredSeconds: z.number().nonnegative(),
  coveredIntervals: z.array(intervalSchema),
  mediaDuration: z.number().nonnegative().nullable(),
  source: z.enum(["asr", "youtube-captions"]),
  language: z.string().nullable(),
  sessionId: z.string().min(1),
  lastSeq: z.number().int().nonnegative(),
});

export type SubtitleMetaPayload = z.infer<typeof subtitleMetaSchema>;

export const tabStatusSchema = z.object({
  mode: z.enum(["idle", "disabled", "playback", "recognize", "youtube-captions", "tab-capture", "closed", "error"]),
  canonicalUrl: z.string().nullable(),
  urlKey: z.string().nullable(),
  coveredSeconds: z.number().nonnegative(),
  cueCount: z.number().int().nonnegative(),
  message: z.string().nullable(),
  recording: z.boolean().default(false),
  canSave: z.boolean().default(false),
  mediaDuration: z.number().nonnegative().nullable().default(null),
  currentTime: z.number().nonnegative().nullable().default(null),
  coverage: z.array(intervalSchema).max(10_000).default([]),
});

export type TabStatus = z.output<typeof tabStatusSchema>;

export const contentRequestSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("resolve-page"), pageUrl: z.string().max(8192) }),
  z.object({
    type: z.literal("save-overlay-layout"),
    offsetX: z.number().min(-1).max(1).optional(),
    offsetY: z.number().min(-1).max(1).optional(),
    heightLines: z.number().int().min(OVERLAY_HEIGHT_LINES.min).max(OVERLAY_HEIGHT_LINES.max).optional(),
  }),
  z.object({ type: z.literal("translate-lines"), texts: z.array(z.string().max(2000)).min(1).max(MAX_TRANSLATE_LINES) }),
  z.object({
    type: z.literal("save-subtitle"),
    urlKey: urlKeySchema,
    meta: subtitleMetaSchema,
    cues: z.array(z.object({ start: z.number().nonnegative(), end: z.number().nonnegative(), text: z.string().min(1).max(2000) })).max(20_000),
    placement: z.enum(["timeline", "append"]),
    expectedRevision: z.number().int().nonnegative(),
    confirmOverwrite: z.boolean().default(false),
  }),
  z.object({ type: z.literal("report-status"), status: tabStatusSchema }),
]);

export type ContentRequest = z.infer<typeof contentRequestSchema>;

export const extensionPageRequestSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("get-tab-status"), tabId: z.number().int() }),
  z.object({ type: z.literal("delete-subtitle"), tabId: z.number().int() }),
  z.object({ type: z.literal("undo-recording"), tabId: z.number().int() }),
  z.object({ type: z.literal("start-recognition"), tabId: z.number().int() }),
  z.object({ type: z.literal("stop-recognition"), tabId: z.number().int() }),
  z.object({ type: z.literal("save-now"), tabId: z.number().int() }),
  z.object({ type: z.literal("seek"), tabId: z.number().int(), time: z.number().nonnegative() }),
  z.object({ type: z.literal("start-tab-capture"), tabId: z.number().int() }),
  z.object({ type: z.literal("stop-tab-capture"), tabId: z.number().int() }),
  z.object({ type: z.literal("check-server") }),
]);

export type ExtensionPageRequest = z.infer<typeof extensionPageRequestSchema>;

export const OFFSCREEN_TARGET = "offscreen";

export type OffscreenCommand =
  | {
      readonly target: typeof OFFSCREEN_TARGET;
      readonly type: "offscreen-start";
      readonly tabId: number;
      readonly streamId: string;
    }
  | { readonly target: typeof OFFSCREEN_TARGET; readonly type: "offscreen-stop"; readonly tabId: number };

export const offscreenEventSchema = z.object({ type: z.literal("offscreen-ended"), tabId: z.number().int(), message: z.string().nullable() });

export type OffscreenEvent = z.infer<typeof offscreenEventSchema>;

export type TabBroadcast =
  | { readonly type: "tab-capture-started" }
  | { readonly type: "tab-capture-partial"; readonly start: number; readonly end: number; readonly text: string }
  | { readonly type: "tab-capture-segment"; readonly start: number; readonly end: number; readonly display: string }
  | { readonly type: "tab-capture-stopped"; readonly message: string | null }
  | { readonly type: "subtitle-changed" }
  | { readonly type: "recognition-start" }
  | { readonly type: "recognition-stop"; readonly discard: boolean }
  | { readonly type: "save-now" }
  | { readonly type: "seek"; readonly time: number };

export interface ContentSettings {
  readonly flushIntervalSeconds: number;
  readonly completionRatio: number;
  readonly overlay: { readonly fontSizePx: number; readonly maxSentences: number; readonly opacity: number; readonly offsetX: number; readonly offsetY: number; readonly heightLines: number };
  readonly features: { readonly autoStart: boolean; readonly useYoutubeCaptions: boolean; readonly recognizeOnlyAtNormalRate: boolean };
}

export type ResolvedPage =
  | { readonly mode: "disabled"; readonly reason: string }
  | {
      readonly mode: "playback" | "recognize";
      readonly urlKey: string;
      readonly canonicalUrl: string;
      readonly cues: Cue[];
      readonly revision: number;
      readonly existing: SubtitleMetaPayload | null;
      readonly settings: ContentSettings;
    };

export interface TranslateLinesResult {
  readonly displays: string[];
}

export interface SaveResult {
  readonly written: boolean;
  readonly revision: number;
  readonly reason: string | null;
}

export type Reply<T> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: string };

export function okReply<T>(value: T): Reply<T> {
  return { ok: true, value };
}

export function errorReply(error: unknown): Reply<never> {
  return { ok: false, error: error instanceof Error ? error.message : String(error) };
}
