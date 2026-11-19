import { OVERLAY_HEIGHT_LINES } from "../shared/overlay-layout";

interface ResizeState {
  readonly pointerId: number;
  readonly startY: number;
  readonly startLines: number;
}

function clampLines(lines: number): number {
  return Math.max(OVERLAY_HEIGHT_LINES.min, Math.min(OVERLAY_HEIGHT_LINES.max, lines));
}

export class CaptionHeightResizer {
  private resize: ResizeState | null = null;

  constructor(
    private readonly handle: HTMLElement,
    private readonly lineHeightPx: () => number,
    private lines: number,
    private readonly onPreview: (lines: number) => void,
    private readonly onCommit: (lines: number) => void,
  ) {
    onPreview(lines);
    handle.addEventListener("pointerdown", (event) => this.begin(event));
    handle.addEventListener("pointermove", (event) => this.move(event));
    handle.addEventListener("pointerup", (event) => this.end(event));
    handle.addEventListener("pointercancel", (event) => this.end(event));
    handle.addEventListener("dblclick", (event) => event.stopPropagation());
  }

  private begin(event: PointerEvent): void {
    event.preventDefault();
    event.stopPropagation();
    if (event.button !== 0 || !event.isTrusted) {
      return;
    }
    this.handle.setPointerCapture(event.pointerId);
    this.resize = { pointerId: event.pointerId, startY: event.clientY, startLines: this.lines };
  }

  private move(event: PointerEvent): void {
    const resize = this.resize;
    if (!resize || resize.pointerId !== event.pointerId) {
      return;
    }
    const lines = clampLines(resize.startLines + Math.round((resize.startY - event.clientY) / this.lineHeightPx()));
    if (lines !== this.lines) {
      this.lines = lines;
      this.onPreview(lines);
    }
  }

  private end(event: PointerEvent): void {
    const resize = this.resize;
    if (!resize || resize.pointerId !== event.pointerId) {
      return;
    }
    this.resize = null;
    if (this.lines !== resize.startLines) {
      this.onCommit(this.lines);
    }
  }
}
