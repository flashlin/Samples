import { bytesToBase64 } from "./base64";
import type { PcmFrame } from "./audio/pcm-tap";

export const CONTENT_STREAM_PORT = "caption-stream";
export const TAB_STREAM_PORT_PREFIX = "caption-tab-stream:";

export type StreamClientMessage =
  | { readonly type: "stream-audio"; readonly start: number; readonly pcm: string }
  | { readonly type: "stream-break" };

export interface StreamSegment {
  readonly start: number;
  readonly end: number;
  readonly display: string;
}

export type StreamServerMessage =
  | { readonly type: "ready" }
  | { readonly type: "partial"; readonly start: number; readonly end: number; readonly text: string }
  | { readonly type: "final"; readonly start: number; readonly end: number; readonly segments: readonly StreamSegment[] }
  | { readonly type: "covered"; readonly start: number; readonly end: number }
  | { readonly type: "error"; readonly message: string }
  | { readonly type: "closed"; readonly message: string | null };

export function audioMessage(frame: PcmFrame): StreamClientMessage {
  return { type: "stream-audio", start: frame.start, pcm: bytesToBase64(new Uint8Array(frame.samples.buffer, frame.samples.byteOffset, frame.samples.byteLength)) };
}

export function isStreamClientMessage(value: unknown): value is StreamClientMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const message = value as Record<string, unknown>;
  if (message.type === "stream-break") {
    return true;
  }
  return message.type === "stream-audio" && typeof message.start === "number" && Number.isFinite(message.start) && message.start >= 0 && typeof message.pcm === "string";
}
