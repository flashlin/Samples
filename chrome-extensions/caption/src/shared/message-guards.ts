import type { OffscreenCommand, TabBroadcast } from "./messages";

type Fields = Record<string, unknown>;

function isRecord(value: unknown): value is Fields {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function isTabBroadcast(value: unknown): value is TabBroadcast {
  if (!isRecord(value)) {
    return false;
  }
  switch (value.type) {
    case "tab-capture-started":
    case "subtitle-deleted":
      return true;
    case "tab-capture-segment":
      return isFiniteNumber(value.start) && isFiniteNumber(value.end) && typeof value.display === "string";
    case "tab-capture-stopped":
      return value.message === null || typeof value.message === "string";
    default:
      return false;
  }
}

function isChunkLimits(value: unknown): boolean {
  return isRecord(value) && isFiniteNumber(value.minSeconds) && isFiniteNumber(value.maxSeconds) && isFiniteNumber(value.silenceSeconds);
}

export function isOffscreenCommand(value: unknown): value is OffscreenCommand {
  if (!isRecord(value) || value.target !== "offscreen" || !Number.isInteger(value.tabId)) {
    return false;
  }
  if (value.type === "offscreen-start") {
    return typeof value.streamId === "string" && value.streamId.length > 0 && isChunkLimits(value.limits);
  }
  return value.type === "offscreen-stop";
}
