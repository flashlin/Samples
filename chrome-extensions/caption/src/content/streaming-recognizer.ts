import { PcmTap } from "../shared/audio/pcm-tap";
import { audioMessage, CONTENT_STREAM_PORT, type StreamClientMessage, type StreamSegment, type StreamServerMessage } from "../shared/stream-protocol";

const NORMAL_PLAYBACK_RATE = 1;

export interface StreamCallbacks {
  readonly partial: (start: number, end: number, text: string) => void;
  readonly final: (start: number, end: number, segments: readonly StreamSegment[]) => void;
  readonly covered: (start: number, end: number) => void;
  readonly problem: (message: string) => void;
}

export interface StreamingOptions {
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

export function deliverStreamMessage(message: StreamServerMessage, callbacks: StreamCallbacks): void {
  switch (message.type) {
    case "partial":
      callbacks.partial(message.start, message.end, message.text);
      return;
    case "final":
      callbacks.final(message.start, message.end, message.segments);
      return;
    case "covered":
      callbacks.covered(message.start, message.end);
      return;
    case "error":
      callbacks.problem(message.message);
      return;
    case "closed":
      if (message.message) {
        callbacks.problem(message.message);
      }
      return;
    case "ready":
      return;
  }
}

export class StreamingRecognizer {
  private tap: PcmTap | null = null;
  private port: chrome.runtime.Port | null = null;
  private readonly events = new AbortController();
  private adTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    private readonly media: HTMLMediaElement,
    private readonly callbacks: StreamCallbacks,
    private readonly options: StreamingOptions,
  ) {}

  start(): void {
    const stream = captureMediaStream(this.media);
    const port = chrome.runtime.connect({ name: CONTENT_STREAM_PORT });
    port.onMessage.addListener((message: StreamServerMessage) => deliverStreamMessage(message, this.callbacks));
    port.onDisconnect.addListener(() => {
      this.port = null;
    });
    this.port = port;
    this.tap = new PcmTap(stream, {
      clock: () => this.media.currentTime,
      onFrame: (frame) => this.post(audioMessage(frame)),
      onDiscontinuity: () => this.post({ type: "stream-break" }),
    });
    const sync = () => this.syncRecording();
    for (const type of ["play", "playing", "pause", "ratechange", "ended", "emptied", "seeking", "seeked"]) {
      this.media.addEventListener(type, sync, { signal: this.events.signal });
    }
    this.adTimer = setInterval(sync, 1000);
    this.syncRecording();
  }

  stop(): void {
    this.events.abort();
    if (this.adTimer !== null) {
      clearInterval(this.adTimer);
    }
    this.tap?.stop();
    this.tap = null;
    this.port?.disconnect();
    this.port = null;
  }

  private post(message: StreamClientMessage): void {
    this.port?.postMessage(message);
  }

  private shouldRecord(): boolean {
    const media = this.media;
    const rateAllowed = !this.options.recognizeOnlyAtNormalRate || media.playbackRate === NORMAL_PLAYBACK_RATE;
    return !media.paused && !media.ended && !media.seeking && rateAllowed && !this.options.isAdPlaying();
  }

  private syncRecording(): void {
    this.tap?.setActive(this.shouldRecord());
  }
}
