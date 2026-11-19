export type SubtitleStatus = "partial" | "complete";

export interface SubtitleProgress {
  readonly status: SubtitleStatus;
  readonly coveredSeconds: number;
  readonly sessionId: string;
  readonly lastSeq: number;
}

export type OverwriteRejection = "empty-candidate" | "stale-sequence" | "confirmation-required";

export type OverwriteDecision = { readonly write: true } | { readonly write: false; readonly reason: OverwriteRejection };

const ACCEPT: OverwriteDecision = { write: true };

function reject(reason: OverwriteRejection): OverwriteDecision {
  return { write: false, reason };
}

export function decideOverwrite(existing: SubtitleProgress | null, candidate: SubtitleProgress, confirmed: boolean): OverwriteDecision {
  if (candidate.status === "partial" && candidate.coveredSeconds <= 0) {
    return reject("empty-candidate");
  }
  if (existing === null) {
    return ACCEPT;
  }
  if (existing.sessionId === candidate.sessionId) {
    return candidate.lastSeq > existing.lastSeq ? ACCEPT : reject("stale-sequence");
  }
  return confirmed ? ACCEPT : reject("confirmation-required");
}
