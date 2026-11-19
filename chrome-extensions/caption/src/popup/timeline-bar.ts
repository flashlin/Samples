import { findGaps, type Interval } from "../shared/progress/coverage";

const MINIMUM_GAP_SECONDS = 1;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;
const PERCENT = 100;

export interface TimelineData {
  readonly duration: number;
  readonly currentTime: number | null;
  readonly coverage: readonly Interval[];
}

export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / SECONDS_PER_HOUR);
  const minutes = Math.floor((whole % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  const secs = String(whole % SECONDS_PER_MINUTE).padStart(2, "0");
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

function percent(seconds: number, duration: number): string {
  return `${(Math.min(Math.max(seconds / duration, 0), 1) * PERCENT).toFixed(3)}%`;
}

export class TimelineBar {
  constructor(
    private readonly bar: HTMLElement,
    private readonly gapList: HTMLElement,
    private readonly onSeek: (seconds: number) => void,
  ) {
    this.bar.addEventListener("click", (event) => {
      const duration = Number(this.bar.dataset.duration);
      if (duration > 0) {
        const rect = this.bar.getBoundingClientRect();
        this.onSeek(((event.clientX - rect.left) / rect.width) * duration);
      }
    });
  }

  render(data: TimelineData | null): void {
    this.bar.hidden = data === null;
    this.gapList.hidden = data === null;
    if (data === null) {
      return;
    }
    this.bar.dataset.duration = String(data.duration);
    this.bar.replaceChildren(...data.coverage.map(([start, end]) => this.span("covered", start, end, data.duration)), ...this.playhead(data));
    this.renderGaps(findGaps(data.coverage, data.duration, MINIMUM_GAP_SECONDS));
  }

  private span(className: string, start: number, end: number, duration: number): HTMLElement {
    const element = document.createElement("div");
    element.className = className;
    element.style.left = percent(start, duration);
    element.style.width = percent(end - start, duration);
    return element;
  }

  private playhead(data: TimelineData): HTMLElement[] {
    if (data.currentTime === null) {
      return [];
    }
    const marker = document.createElement("div");
    marker.className = "playhead";
    marker.style.left = percent(data.currentTime, data.duration);
    return [marker];
  }

  private renderGaps(gaps: Interval[]): void {
    if (gaps.length === 0) {
      this.gapList.textContent = "No gaps";
      return;
    }
    const label = document.createElement("span");
    label.textContent = "Gaps: ";
    const links = gaps.map(([start, end]) => {
      const link = document.createElement("button");
      link.type = "button";
      link.className = "gap";
      link.textContent = `${formatClock(start)}–${formatClock(end)}`;
      link.addEventListener("click", () => this.onSeek(start));
      return link;
    });
    this.gapList.replaceChildren(label, ...links);
  }
}
