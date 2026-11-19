import { MediaTimeMapper } from "../shared/audio/media-time-mapper";
import type { Cue, SegmentSink } from "../shared/cue/cue";


const MEDIA_SAMPLE_MILLISECONDS = 250;
const MILLISECONDS_PER_SECOND = 1000;

function wallSeconds(): number {
  return Date.now() / MILLISECONDS_PER_SECOND;
}

export class TabCaptureReceiver {
  private readonly mapper = new MediaTimeMapper();
  private readonly startedAt = wallSeconds();
  private readonly timer: ReturnType<typeof setInterval>;
  private latest: Cue | null = null;

  constructor(
    private readonly currentMedia: () => HTMLMediaElement | null,
    private readonly sink: SegmentSink,
  ) {
    this.timer = setInterval(() => this.sampleMediaTime(), MEDIA_SAMPLE_MILLISECONDS);
  }

  get latestCue(): Cue | null {
    return this.latest;
  }

  receivePartial(startWall: number, endWall: number, text: string): void {
    const start = this.toTimeline(startWall);
    this.latest = { start, end: Math.max(start, this.toTimeline(endWall)), text };
  }

  receive(startWall: number, endWall: number, display: string): void {
    const start = this.toTimeline(startWall);
    const end = Math.max(start, this.toTimeline(endWall));
    this.sink(start, end, display);
    if (display) {
      this.latest = { start, end, text: display };
    }
  }

  stop(): void {
    clearInterval(this.timer);
  }

  private sampleMediaTime(): void {
    const media = this.currentMedia();
    if (media) {
      this.mapper.record({ wallSeconds: wallSeconds(), mediaSeconds: media.currentTime, playing: !media.paused, playbackRate: media.playbackRate });
    }
  }

  private toTimeline(wall: number): number {
    return (this.currentMedia() ? this.mapper.toMediaTime(wall) : null) ?? wall - this.startedAt;
  }
}
