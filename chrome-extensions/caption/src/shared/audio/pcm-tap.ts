export const STREAM_SAMPLE_RATE = 16_000;
const FRAME_SAMPLES = 2048;
const RESYNC_THRESHOLD_SECONDS = 0.3;
const INT16_MAX = 0x7fff;

export interface PcmFrame {
  readonly start: number;
  readonly samples: Int16Array;
}

export interface PcmTapOptions {
  readonly clock: () => number;
  readonly onFrame: (frame: PcmFrame) => void;
  readonly onDiscontinuity: () => void;
}

function toInt16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index++) {
    const value = Math.max(-1, Math.min(1, input[index]!));
    output[index] = Math.round(value * INT16_MAX);
  }
  return output;
}

export class PcmTap {
  private readonly context = new AudioContext({ sampleRate: STREAM_SAMPLE_RATE });
  private readonly processor: ScriptProcessorNode;
  private active = false;
  private nextStart: number | null = null;

  constructor(
    stream: MediaStream,
    private readonly options: PcmTapOptions,
  ) {
    const source = this.context.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    this.processor = this.context.createScriptProcessor(FRAME_SAMPLES, 1, 1);
    const mute = this.context.createGain();
    mute.gain.value = 0;
    source.connect(this.processor);
    this.processor.connect(mute);
    mute.connect(this.context.destination);
    this.processor.onaudioprocess = (event) => this.capture(event.inputBuffer.getChannelData(0));
  }

  setActive(active: boolean): void {
    if (active) {
      void this.context.resume().catch(() => undefined);
    }
    if (this.active && !active) {
      this.nextStart = null;
      this.options.onDiscontinuity();
    }
    this.active = active;
  }

  stop(): void {
    this.setActive(false);
    this.processor.onaudioprocess = null;
    void this.context.close().catch(() => undefined);
  }

  private capture(input: Float32Array): void {
    if (!this.active) {
      return;
    }
    const seconds = input.length / STREAM_SAMPLE_RATE;
    const measuredStart = Math.max(0, this.options.clock() - seconds);
    if (this.nextStart !== null && Math.abs(measuredStart - this.nextStart) > RESYNC_THRESHOLD_SECONDS) {
      this.options.onDiscontinuity();
      this.nextStart = null;
    }
    const start = this.nextStart ?? measuredStart;
    this.nextStart = start + seconds;
    this.options.onFrame({ start, samples: toInt16(input) });
  }
}
