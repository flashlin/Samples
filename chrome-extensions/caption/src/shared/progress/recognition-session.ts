import type { Cue } from "../cue/cue";
import { mergeCues } from "../cue/cue-merger";
import { serializeSrt } from "../srt/srt-codec";
import type { SubtitleMetaPayload } from "../messages";
import { addInterval, coveredSeconds, isRecognitionComplete, type Interval } from "./coverage";
import { decideOverwrite, type SubtitleProgress } from "./overwrite-policy";

const RECORDER_GAP_TOLERANCE_SECONDS = 0.1;

export interface SessionIdentity {
  readonly sessionId: string;
  readonly canonicalUrl: string;
  readonly source: SubtitleMetaPayload["source"];
  readonly completionRatio: number;
}

export interface SessionSnapshot {
  readonly meta: SubtitleMetaPayload;
  readonly srt: string;
}

export class RecognitionSession {
  private cues: Cue[] = [];
  private coverage: Interval[] = [];
  private mediaDuration: number | null = null;
  private ended = false;
  private seq = 0;
  private dirty = false;
  private source: SessionIdentity["source"];

  constructor(
    private readonly identity: SessionIdentity,
    private existing: SubtitleMetaPayload | null,
  ) {
    this.source = identity.source;
  }

  useSource(source: SessionIdentity["source"]): void {
    if (source !== this.source) {
      this.source = source;
      this.dirty = true;
    }
  }

  get liveCues(): readonly Cue[] {
    return this.cues;
  }

  get liveCoverage(): readonly Interval[] {
    return this.coverage;
  }

  get coveredSeconds(): number {
    return coveredSeconds(this.coverage);
  }

  get isComplete(): boolean {
    return isRecognitionComplete({
      coveredSeconds: this.coveredSeconds,
      mediaDuration: this.mediaDuration,
      ended: this.ended,
      completionRatio: this.identity.completionRatio,
    });
  }

  recordSegment(start: number, end: number, text: string): void {
    if (!(end > start)) {
      return;
    }
    this.coverage = addInterval(this.coverage, [start, end], RECORDER_GAP_TOLERANCE_SECONDS);
    const trimmed = text.trim();
    if (trimmed) {
      this.cues = mergeCues(this.cues, [{ start, end, text: trimmed }]);
    }
    this.dirty = true;
  }

  updateMedia(duration: number | null, ended: boolean): void {
    this.mediaDuration = duration !== null && Number.isFinite(duration) && duration > 0 ? duration : null;
    if (ended && !this.ended) {
      this.ended = true;
      this.dirty = true;
    }
  }

  private progress(lastSeq: number): SubtitleProgress {
    return { status: this.isComplete ? "complete" : "partial", coveredSeconds: this.coveredSeconds, sessionId: this.identity.sessionId, lastSeq };
  }

  takeSnapshotIfWorthSaving(): SessionSnapshot | null {
    if (!this.dirty) {
      return null;
    }
    const candidate = this.progress(this.seq + 1);
    if (!decideOverwrite(this.existing, candidate).write) {
      return null;
    }
    this.seq = candidate.lastSeq;
    this.dirty = false;
    return {
      meta: {
        canonicalUrl: this.identity.canonicalUrl,
        status: candidate.status,
        coveredSeconds: candidate.coveredSeconds,
        coveredIntervals: this.coverage.map(([start, end]) => [start, end]),
        mediaDuration: this.mediaDuration,
        source: this.source,
        language: null,
        sessionId: this.identity.sessionId,
        lastSeq: candidate.lastSeq,
      },
      srt: serializeSrt(this.cues),
    };
  }

  acknowledgeSaved(snapshot: SessionSnapshot): void {
    this.existing = snapshot.meta;
  }

  markDirty(): void {
    this.dirty = true;
  }
}
