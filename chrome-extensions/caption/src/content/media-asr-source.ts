import { AudioTap } from "../shared/audio/audio-tap";
import type { ChunkLimits } from "../shared/audio/chunk-cut-policy";
import type { TranscriptionQueue } from "./transcription-queue";

const NORMAL_PLAYBACK_RATE = 1;

export interface MediaAsrOptions {
  readonly limits: ChunkLimits;
  readonly interimSeconds: number;
  readonly recognizeOnlyAtNormalRate: boolean;
  readonly isAdPlaying: () => boolean;
}

export class CrossOriginMediaError extends Error {}

function captureMediaStream(media: HTMLMediaElement): MediaStream {
  try {
    return media.captureStream();
  } catch (error) {
    if (error instanceof DOMException && error.name === "SecurityError") {
      throw new CrossOriginMediaError("This media is cross-origin and cannot be captured directly");
    }
    throw error;
  }
}

export class MediaAsrSource {
  private tap: AudioTap | null = null;
  private readonly controller = new AbortController();

  constructor(
    private readonly media: HTMLMediaElement,
    private readonly queue: TranscriptionQueue,
    private readonly options: MediaAsrOptions,
  ) {}

  start(): void {
    const stream = captureMediaStream(this.media);
    this.tap = new AudioTap(stream, {
      limits: this.options.limits,
      clock: () => this.media.currentTime,
      onChunk: (chunk) => {
        if (!this.options.isAdPlaying()) {
          this.queue.enqueue(chunk);
        }
      },
      interim:
        this.options.interimSeconds > 0
          ? {
              everySeconds: this.options.interimSeconds,
              onInterim: (chunk) => {
                if (!this.options.isAdPlaying()) {
                  this.queue.enqueueInterim(chunk);
                }
              },
            }
          : undefined,
    });
    const signal = this.controller.signal;
    const sync = () => this.syncRecording();
    for (const type of ["play", "playing", "pause", "ratechange", "ended", "emptied", "seeked"]) {
      this.media.addEventListener(type, sync, { signal });
    }
    this.media.addEventListener("seeking", () => this.tap?.pause(this.tap.lastObservedClock), { signal });
    this.syncRecording();
  }

  stop(): void {
    this.controller.abort();
    this.tap?.stop();
    this.tap = null;
  }

  private shouldRecord(): boolean {
    const media = this.media;
    const rateAllowed = !this.options.recognizeOnlyAtNormalRate || media.playbackRate === NORMAL_PLAYBACK_RATE;
    return !media.paused && !media.ended && !media.seeking && rateAllowed && !this.options.isAdPlaying();
  }

  private syncRecording(): void {
    if (!this.tap) {
      return;
    }
    if (this.shouldRecord()) {
      this.tap.resume();
    } else if (this.tap.isRecording) {
      this.tap.pause();
    }
  }
}
