import type { Cue, SegmentSink } from "../shared/cue/cue";
import { askBackground } from "./background-client";
import { hideNativeYoutubeCaptions, isYoutubeAdPlaying, readVisibleYoutubeCaption } from "./youtube";

const POLL_MILLISECONDS = 200;
const MAX_CONTINUOUS_STEP_SECONDS = 1.5;
const TRANSLATE_DEBOUNCE_MILLISECONDS = 300;
const NO_CAPTION_GIVE_UP_SECONDS = 20;

interface OpenCue {
  readonly start: number;
  text: string;
}

export class YoutubeCaptionReader {
  private timer: ReturnType<typeof setInterval> | null = null;
  private translateTimer: ReturnType<typeof setTimeout> | null = null;
  private previousTime: number | null = null;
  private open: OpenCue | null = null;
  private readonly displays = new Map<string, string>();
  private restoreNativeCaptions: (() => void) | null = null;
  private sawCaptionText = false;
  private playedSecondsWithoutCaptions = 0;

  constructor(
    private readonly media: HTMLMediaElement,
    private readonly sink: SegmentSink,
    private readonly onNoCaptions: () => void,
  ) {}

  get currentCue(): Cue | null {
    const open = this.open;
    return open ? { start: open.start, end: this.media.currentTime, text: this.displays.get(open.text) ?? open.text } : null;
  }

  start(): void {
    this.restoreNativeCaptions = hideNativeYoutubeCaptions();
    this.timer = setInterval(() => this.poll(), POLL_MILLISECONDS);
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.translateTimer !== null) {
      clearTimeout(this.translateTimer);
    }
    this.closeOpenCue(this.media.currentTime);
    this.restoreNativeCaptions?.();
    this.restoreNativeCaptions = null;
  }

  private poll(): void {
    const now = this.media.currentTime;
    const playing = !this.media.paused && !isYoutubeAdPlaying();
    const previous = this.previousTime;
    this.previousTime = playing ? now : null;
    if (!playing) {
      this.closeOpenCue(previous ?? now);
      return;
    }
    const text = readVisibleYoutubeCaption();
    this.sawCaptionText ||= text.length > 0;
    if (previous !== null) {
      const step = now - previous;
      if (step < 0 || step > MAX_CONTINUOUS_STEP_SECONDS) {
        this.closeOpenCue(previous);
      } else if (this.sawCaptionText) {
        this.sink(previous, now, "");
      } else if (this.gaveUpWaitingForCaptions(step)) {
        return;
      }
    }
    this.followVisibleText(text, now);
  }

  private gaveUpWaitingForCaptions(playedSeconds: number): boolean {
    this.playedSecondsWithoutCaptions += playedSeconds;
    if (this.playedSecondsWithoutCaptions < NO_CAPTION_GIVE_UP_SECONDS) {
      return false;
    }
    this.stop();
    this.onNoCaptions();
    return true;
  }

  private followVisibleText(text: string, now: number): void {
    const open = this.open;
    if (open && text === open.text) {
      return;
    }
    if (open && text.startsWith(open.text)) {
      open.text = text;
      this.scheduleTranslation(text);
      return;
    }
    this.closeOpenCue(now);
    if (text) {
      this.open = { start: now, text };
      this.scheduleTranslation(text);
    }
  }

  private closeOpenCue(end: number): void {
    const open = this.open;
    this.open = null;
    if (!open || !(end > open.start)) {
      return;
    }
    const cached = this.displays.get(open.text);
    if (cached !== undefined) {
      this.sink(open.start, end, cached);
      return;
    }
    void this.translate(open.text).then((display) => this.sink(open.start, end, display));
  }

  private scheduleTranslation(text: string): void {
    if (this.translateTimer !== null) {
      clearTimeout(this.translateTimer);
    }
    this.translateTimer = setTimeout(() => void this.translate(text), TRANSLATE_DEBOUNCE_MILLISECONDS);
  }

  private async translate(text: string): Promise<string> {
    const cached = this.displays.get(text);
    if (cached !== undefined) {
      return cached;
    }
    try {
      const { displays } = await askBackground({ type: "translate-lines", texts: [text] });
      const display = displays[0] ?? text;
      this.displays.set(text, display);
      return display;
    } catch {
      return text;
    }
  }
}
