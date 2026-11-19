export type SubtitleStatus = "partial" | "complete";

export interface SubtitleProgress {
  readonly status: SubtitleStatus;
  readonly coveredSeconds: number;
  readonly sessionId: string;
  readonly lastSeq: number;
}

export type OverwriteRejection = "empty-candidate" | "existing-complete" | "stale-sequence" | "not-ahead";

export type OverwriteDecision = { readonly write: true } | { readonly write: false; readonly reason: OverwriteRejection };

const ACCEPT: OverwriteDecision = { write: true };

function reject(reason: OverwriteRejection): OverwriteDecision {
  return { write: false, reason };
}

export function decideOverwrite(existing: SubtitleProgress | null, candidate: SubtitleProgress): OverwriteDecision {
  if (candidate.status === "partial" && candidate.coveredSeconds <= 0) {
    return reject("empty-candidate");
  }
  if (existing === null) {
    return ACCEPT;
  }
  if (existing.status === "complete") {
    return reject("existing-complete");
  }
  const sameSession = existing.sessionId === candidate.sessionId;
  if (sameSession && candidate.lastSeq <= existing.lastSeq) {
    return reject("stale-sequence");
  }
  if (sameSession || candidate.status === "complete") {
    return ACCEPT;
  }
  return candidate.coveredSeconds > existing.coveredSeconds ? ACCEPT : reject("not-ahead");
}
