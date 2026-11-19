import { rootMeanSquare, shouldCutChunk, SILENCE_RMS_THRESHOLD, type ChunkLimits } from "./chunk-cut-policy";

const TICK_MILLISECONDS = 100;
const MILLISECONDS_PER_SECOND = 1000;
const MIN_CHUNK_SECONDS = 0.15;
const RECORDER_MIME_TYPE = "audio/webm;codecs=opus";
const ANALYSER_FFT_SIZE = 2048;
const RECORDER_TIMESLICE_MILLISECONDS = 250;

export interface AudioChunk {
  readonly blob: Blob;
  readonly start: number;
  readonly end: number;
  readonly silent: boolean;
}

export interface AudioTapOptions {
  readonly limits: ChunkLimits;
  readonly clock: () => number;
  readonly onChunk: (chunk: AudioChunk) => void;
  readonly interim?: { readonly everySeconds: number; readonly onInterim: (chunk: AudioChunk) => void };
}

interface ActiveChunk {
  readonly recorder: MediaRecorder;
  readonly start: number;
  readonly parts: Blob[];
  peakRms: number;
  trailingSilenceSeconds: number;
  lastInterimAt: number;
}

export class AudioTap {
  private readonly audioOnly: MediaStream;
  private readonly context: AudioContext;
  private readonly analyser: AnalyserNode;
  private readonly samples: Float32Array<ArrayBuffer>;
  private active: ActiveChunk | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastTickAt = 0;
  private lastClock = 0;

  constructor(
    stream: MediaStream,
    private readonly options: AudioTapOptions,
  ) {
    this.audioOnly = new MediaStream(stream.getAudioTracks());
    this.context = new AudioContext();
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = ANALYSER_FFT_SIZE;
    this.context.createMediaStreamSource(this.audioOnly).connect(this.analyser);
    this.samples = new Float32Array(this.analyser.fftSize);
  }

  get isRecording(): boolean {
    return this.active !== null;
  }

  get lastObservedClock(): number {
    return this.lastClock;
  }

  get canDetectSilence(): boolean {
    return this.context.state === "running";
  }

  resume(): void {
    void this.context.resume().catch(() => undefined);
    if (this.active) {
      return;
    }
    this.active = this.beginChunk();
    this.lastTickAt = performance.now();
    this.timer ??= setInterval(() => this.tick(), TICK_MILLISECONDS);
  }

  pause(endAt?: number): void {
    this.finishChunk(endAt);
  }

  stop(): void {
    this.finishChunk();
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    void this.context.close().catch(() => undefined);
  }

  private beginChunk(): ActiveChunk {
    const recorder = new MediaRecorder(this.audioOnly, { mimeType: RECORDER_MIME_TYPE });
    const start = this.options.clock();
    const chunk: ActiveChunk = { recorder, start, parts: [], peakRms: 0, trailingSilenceSeconds: 0, lastInterimAt: start };
    recorder.ondataavailable = (event) => chunk.parts.push(event.data);
    recorder.start(RECORDER_TIMESLICE_MILLISECONDS);
    this.lastClock = chunk.start;
    return chunk;
  }

  private finishChunk(endAt?: number): void {
    const chunk = this.active;
    if (!chunk) {
      return;
    }
    this.active = null;
    const end = endAt ?? this.options.clock();
    const silent = this.canDetectSilence && chunk.peakRms < SILENCE_RMS_THRESHOLD;
    chunk.recorder.onstop = () => {
      if (end - chunk.start >= MIN_CHUNK_SECONDS) {
        this.options.onChunk({ blob: new Blob(chunk.parts, { type: RECORDER_MIME_TYPE }), start: chunk.start, end, silent });
      }
    };
    if (chunk.recorder.state !== "inactive") {
      chunk.recorder.stop();
    }
  }

  private measureRms(): number {
    this.analyser.getFloatTimeDomainData(this.samples);
    return rootMeanSquare(this.samples);
  }

  private tick(): void {
    const chunk = this.active;
    const now = performance.now();
    const tickSeconds = (now - this.lastTickAt) / MILLISECONDS_PER_SECOND;
    this.lastTickAt = now;
    if (!chunk) {
      return;
    }
    this.lastClock = this.options.clock();
    if (this.canDetectSilence) {
      const rms = this.measureRms();
      chunk.peakRms = Math.max(chunk.peakRms, rms);
      chunk.trailingSilenceSeconds = rms < SILENCE_RMS_THRESHOLD ? chunk.trailingSilenceSeconds + tickSeconds : 0;
    }
    const state = { elapsedSeconds: this.lastClock - chunk.start, trailingSilenceSeconds: chunk.trailingSilenceSeconds };
    if (shouldCutChunk(state, this.options.limits)) {
      this.finishChunk(this.lastClock);
      this.active = this.beginChunk();
      return;
    }
    this.emitInterimWhenDue(chunk);
  }

  private emitInterimWhenDue(chunk: ActiveChunk): void {
    const interim = this.options.interim;
    const heardSpeech = !this.canDetectSilence || chunk.peakRms >= SILENCE_RMS_THRESHOLD;
    if (!interim || !heardSpeech || chunk.parts.length === 0 || this.lastClock - chunk.lastInterimAt < interim.everySeconds) {
      return;
    }
    chunk.lastInterimAt = this.lastClock;
    interim.onInterim({ blob: new Blob(chunk.parts, { type: RECORDER_MIME_TYPE }), start: chunk.start, end: this.lastClock, silent: false });
  }
}
