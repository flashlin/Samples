import type { AudioChunk } from "../shared/audio/audio-tap";
import { blobToBase64 } from "../shared/base64";
import { askBackground } from "./background-client";

export type SegmentSink = (start: number, end: number, display: string) => void;

export interface QueueLimits {
  readonly maxBacklogChunks: number;
  readonly maxRetries: number;
}

export interface QueueSinks {
  readonly final: SegmentSink;
  readonly interim: SegmentSink;
  readonly error: (error: unknown) => void;
}

export class TranscriptionQueue {
  private readonly pendingFinals: AudioChunk[] = [];
  private pendingInterim: AudioChunk | null = null;
  private latestFinalStart = Number.NEGATIVE_INFINITY;
  private running = false;
  private closed = false;

  constructor(
    private readonly limits: QueueLimits,
    private readonly sinks: QueueSinks,
  ) {}

  enqueue(chunk: AudioChunk): void {
    if (this.closed) {
      return;
    }
    this.latestFinalStart = Math.max(this.latestFinalStart, chunk.start);
    if (this.pendingInterim && this.pendingInterim.start <= chunk.start) {
      this.pendingInterim = null;
    }
    if (chunk.silent) {
      this.sinks.final(chunk.start, chunk.end, "");
      return;
    }
    this.pendingFinals.push(chunk);
    while (this.pendingFinals.length > this.limits.maxBacklogChunks) {
      this.pendingFinals.shift();
    }
    void this.drain();
  }

  enqueueInterim(chunk: AudioChunk): void {
    if (this.closed || chunk.start <= this.latestFinalStart) {
      return;
    }
    this.pendingInterim = chunk;
    void this.drain();
  }

  close(): void {
    this.closed = true;
    this.pendingFinals.length = 0;
    this.pendingInterim = null;
  }

  private nextJob(): { chunk: AudioChunk; interim: boolean } | null {
    const final = this.pendingFinals.shift();
    if (final) {
      return { chunk: final, interim: false };
    }
    const interim = this.pendingInterim;
    this.pendingInterim = null;
    return interim ? { chunk: interim, interim: true } : null;
  }

  private async drain(): Promise<void> {
    if (this.running) {
      return;
    }
    this.running = true;
    try {
      for (let job = this.nextJob(); job && !this.closed; job = this.nextJob()) {
        await (job.interim ? this.transcribeInterim(job.chunk) : this.transcribeFinal(job.chunk));
      }
    } finally {
      this.running = false;
    }
  }

  private async request(chunk: AudioChunk, interim: boolean): Promise<string> {
    const audioBase64 = await blobToBase64(chunk.blob);
    const result = await askBackground({ type: "transcribe", audioBase64, mimeType: chunk.blob.type, interim });
    return result.display;
  }

  private async transcribeInterim(chunk: AudioChunk): Promise<void> {
    try {
      const display = await this.request(chunk, true);
      if (!this.closed && chunk.start > this.latestFinalStart) {
        this.sinks.interim(chunk.start, chunk.end, display);
      }
    } catch {
      return;
    }
  }

  private async transcribeFinal(chunk: AudioChunk): Promise<void> {
    for (let attempt = 0; attempt <= this.limits.maxRetries; attempt++) {
      try {
        const display = await this.request(chunk, false);
        if (!this.closed) {
          this.sinks.final(chunk.start, chunk.end, display);
        }
        return;
      } catch (error) {
        if (attempt === this.limits.maxRetries) {
          this.sinks.error(error);
        }
      }
    }
  }
}
