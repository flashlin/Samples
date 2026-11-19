export interface MediaTimeSample {
  readonly wallSeconds: number;
  readonly mediaSeconds: number;
  readonly playing: boolean;
  readonly playbackRate: number;
}

const DEFAULT_CAPACITY = 2400;

export class MediaTimeMapper {
  private readonly samples: MediaTimeSample[] = [];

  constructor(private readonly capacity = DEFAULT_CAPACITY) {}

  record(sample: MediaTimeSample): void {
    this.samples.push(sample);
    if (this.samples.length > this.capacity) {
      this.samples.shift();
    }
  }

  toMediaTime(wallSeconds: number): number | null {
    let match: MediaTimeSample | undefined;
    for (let index = this.samples.length - 1; index >= 0; index--) {
      const sample = this.samples[index]!;
      if (sample.wallSeconds <= wallSeconds) {
        match = sample;
        break;
      }
    }
    match ??= this.samples[0];
    if (!match) {
      return null;
    }
    const drift = match.playing ? (wallSeconds - match.wallSeconds) * match.playbackRate : 0;
    return Math.max(0, match.mediaSeconds + drift);
  }
}
