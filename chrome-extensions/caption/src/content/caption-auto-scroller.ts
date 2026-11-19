const START_DELAY_MILLISECONDS = 1200;
const SECONDS_PER_LINE = 1.5;
const MILLISECONDS_PER_SECOND = 1000;

export class CaptionAutoScroller {
  private startTimer: ReturnType<typeof setTimeout> | null = null;
  private frame: number | null = null;
  private position = 0;
  private pausedByUser = false;

  constructor(
    private readonly element: HTMLElement,
    private readonly lineHeightPx: () => number,
  ) {
    element.addEventListener("wheel", () => this.pauseByUser(), { passive: true });
  }

  showNewText(): void {
    this.stop();
    this.pausedByUser = false;
    this.position = 0;
    this.element.scrollTop = 0;
    this.startTimer = setTimeout(() => this.scrollFrom(performance.now()), START_DELAY_MILLISECONDS);
  }

  showGrownText(): void {
    if (!this.pausedByUser && this.startTimer === null && this.frame === null) {
      this.scrollFrom(performance.now());
    }
  }

  stop(): void {
    if (this.startTimer !== null) {
      clearTimeout(this.startTimer);
      this.startTimer = null;
    }
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  private pauseByUser(): void {
    this.pausedByUser = true;
    this.stop();
  }

  private scrollFrom(previous: number): void {
    this.startTimer = null;
    this.frame = requestAnimationFrame((now) => {
      this.frame = null;
      const bottom = this.element.scrollHeight - this.element.clientHeight;
      if (this.pausedByUser || this.position >= bottom) {
        return;
      }
      const pixelsPerMillisecond = this.lineHeightPx() / (SECONDS_PER_LINE * MILLISECONDS_PER_SECOND);
      this.position = Math.min(bottom, this.position + (now - previous) * pixelsPerMillisecond);
      this.element.scrollTop = this.position;
      this.scrollFrom(now);
    });
  }
}
