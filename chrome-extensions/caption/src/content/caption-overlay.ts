import type { ContentSettings } from "../shared/messages";

const POSITION_REFRESH_MILLISECONDS = 250;
const BOTTOM_OFFSET_RATIO = 0.14;
const VIEWPORT_BOTTOM_MARGIN_PX = 48;
const VIEWPORT_EDGE_MARGIN_PX = 8;
const MESSAGE_SECONDS = 6;
const MILLISECONDS_PER_SECOND = 1000;
const MAX_OFFSET_RATIO = 1;

const STYLE = `
:host { all: initial; }
.box {
  position: fixed; transform: translate(-50%, -100%); box-sizing: border-box;
  min-width: min(420px, 80vw); max-width: min(80vw, 960px); min-height: calc(var(--caption-font-size) * 1.4 + 12px);
  display: flex; align-items: center; gap: 8px; padding: 6px 10px 6px 14px; border-radius: 8px;
  background: rgba(0, 0, 0, var(--caption-opacity)); color: #fff; z-index: 2147483647;
  font: 500 var(--caption-font-size)/1.4 system-ui, -apple-system, "PingFang TC", "Noto Sans TC", sans-serif;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35); pointer-events: auto; cursor: grab; user-select: none; touch-action: none;
}
.box.dragging { cursor: grabbing; }
.content { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.prompt { display: none; align-items: center; justify-content: center; flex-wrap: wrap; gap: 8px; font-size: 0.7em; cursor: default; }
.prompt.visible { display: flex; }
.prompt button {
  all: unset; cursor: pointer; padding: 3px 10px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.5); color: #fff;
}
.prompt button.primary { background: #1f6feb; border-color: #1f6feb; }
.prompt button:hover, .prompt button:focus-visible { filter: brightness(1.15); }
.text {
  white-space: pre-line; overflow: hidden; display: -webkit-box; -webkit-box-orient: vertical;
  -webkit-line-clamp: 4; text-align: center;
}
.message { font-size: 0.75em; opacity: 0.85; font-style: italic; }
.close {
  all: unset; cursor: pointer; font-size: 14px; line-height: 1; padding: 2px 4px; border-radius: 4px; opacity: 0.7; color: #fff;
}
.close:hover, .close:focus-visible { opacity: 1; background: rgba(255, 255, 255, 0.2); }
`;

export interface OverlayOffset {
  readonly x: number;
  readonly y: number;
}

export interface OverlayCallbacks {
  readonly onClose: () => void;
  readonly onMoved: (offset: OverlayOffset) => void;
}

interface DragState {
  readonly pointerId: number;
  readonly startX: number;
  readonly startY: number;
  readonly startOffset: OverlayOffset;
}

function clampRatio(value: number): number {
  return Math.max(-MAX_OFFSET_RATIO, Math.min(MAX_OFFSET_RATIO, value));
}

export class CaptionOverlay {
  private readonly host = document.createElement("div");
  private readonly box: HTMLDivElement;
  private readonly text: HTMLDivElement;
  private readonly prompt: HTMLDivElement;
  private anchor: HTMLMediaElement | null = null;
  private readonly positionTimer: ReturnType<typeof setInterval>;
  private messageTimer: ReturnType<typeof setTimeout> | null = null;
  private captionText = "";
  private offset: OverlayOffset;
  private drag: DragState | null = null;
  private readonly onFullscreenChange = () => this.reparent();

  constructor(
    settings: ContentSettings["overlay"],
    private readonly callbacks: OverlayCallbacks,
  ) {
    this.offset = { x: settings.offsetX, y: settings.offsetY };
    this.host.setAttribute("data-caption-extension", "");
    const shadow = this.host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = STYLE;
    this.box = document.createElement("div");
    this.box.className = "box";
    this.box.title = "Drag to move, double-click to reset position";
    this.box.style.setProperty("--caption-font-size", `${settings.fontSizePx}px`);
    this.box.style.setProperty("--caption-opacity", String(settings.opacity));
    this.text = document.createElement("div");
    this.text.className = "text";
    this.prompt = document.createElement("div");
    this.prompt.className = "prompt";
    this.prompt.addEventListener("pointerdown", (event) => event.stopPropagation());
    const content = document.createElement("div");
    content.className = "content";
    content.append(this.text, this.prompt);
    this.box.append(content, this.createCloseButton());
    this.listenForDrag();
    shadow.append(style, this.box);
    this.reparent();
    document.addEventListener("fullscreenchange", this.onFullscreenChange);
    this.positionTimer = setInterval(() => this.reposition(), POSITION_REFRESH_MILLISECONDS);
  }

  attachTo(media: HTMLMediaElement | null): void {
    this.anchor = media;
    this.reposition();
  }

  showCaption(text: string | null): void {
    this.captionText = text ?? "";
    if (this.messageTimer === null) {
      this.render(this.captionText, false);
    }
  }

  showMessage(message: string): void {
    this.render(message, true);
    if (this.messageTimer !== null) {
      clearTimeout(this.messageTimer);
    }
    this.messageTimer = setTimeout(() => {
      this.messageTimer = null;
      this.render(this.captionText, false);
    }, MESSAGE_SECONDS * MILLISECONDS_PER_SECOND);
  }

  askConfirmation(question: string, confirmLabel: string, declineLabel: string): Promise<boolean> {
    return new Promise((resolve) => {
      const label = document.createElement("span");
      label.textContent = question;
      const answer = (confirmed: boolean) => {
        this.prompt.replaceChildren();
        this.prompt.classList.remove("visible");
        this.reposition();
        resolve(confirmed);
      };
      this.prompt.replaceChildren(label, this.promptButton(confirmLabel, true, () => answer(true)), this.promptButton(declineLabel, false, () => answer(false)));
      this.prompt.classList.add("visible");
      this.reposition();
    });
  }

  private promptButton(text: string, primary: boolean, onClick: () => void): HTMLButtonElement {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.classList.toggle("primary", primary);
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      if (event.isTrusted) {
        onClick();
      }
    });
    return button;
  }

  destroy(): void {
    clearInterval(this.positionTimer);
    if (this.messageTimer !== null) {
      clearTimeout(this.messageTimer);
    }
    document.removeEventListener("fullscreenchange", this.onFullscreenChange);
    this.host.remove();
  }

  private createCloseButton(): HTMLButtonElement {
    const close = document.createElement("button");
    close.className = "close";
    close.type = "button";
    close.textContent = "✕";
    close.title = "Close captions";
    close.setAttribute("aria-label", "Close captions");
    close.addEventListener("pointerdown", (event) => event.stopPropagation());
    close.addEventListener("click", (event) => {
      event.stopPropagation();
      if (event.isTrusted) {
        this.callbacks.onClose();
      }
    });
    return close;
  }

  private listenForDrag(): void {
    this.box.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      this.box.setPointerCapture(event.pointerId);
      this.box.classList.add("dragging");
      this.drag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, startOffset: this.offset };
    });
    this.box.addEventListener("pointermove", (event) => {
      const drag = this.drag;
      if (!drag || drag.pointerId !== event.pointerId) {
        return;
      }
      const frame = this.frame();
      this.offset = {
        x: clampRatio(drag.startOffset.x + (event.clientX - drag.startX) / frame.width),
        y: clampRatio(drag.startOffset.y + (event.clientY - drag.startY) / frame.height),
      };
      this.reposition();
    });
    const endDrag = (event: PointerEvent) => {
      if (!this.drag || this.drag.pointerId !== event.pointerId) {
        return;
      }
      this.drag = null;
      this.box.classList.remove("dragging");
      this.callbacks.onMoved(this.offset);
    };
    this.box.addEventListener("pointerup", endDrag);
    this.box.addEventListener("pointercancel", endDrag);
    this.box.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      this.offset = { x: 0, y: 0 };
      this.reposition();
      this.callbacks.onMoved(this.offset);
    });
    for (const type of ["click", "mousedown", "mouseup"]) {
      this.box.addEventListener(type, (event) => event.stopPropagation());
    }
  }

  private render(text: string, isMessage: boolean): void {
    if (this.text.textContent === text && this.text.classList.contains("message") === isMessage) {
      return;
    }
    this.text.textContent = text;
    this.text.classList.toggle("message", isMessage);
    this.reposition();
  }

  private reparent(): void {
    const fullscreen = document.fullscreenElement;
    const container = fullscreen && !(fullscreen instanceof HTMLMediaElement) ? fullscreen : document.documentElement;
    if (this.host.parentNode !== container) {
      container.append(this.host);
    }
  }

  private visibleAnchorRect(): DOMRect | null {
    const rect = this.anchor?.getBoundingClientRect();
    const visible = rect !== undefined && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
    return visible ? rect : null;
  }

  private frame(): { width: number; height: number } {
    const rect = this.visibleAnchorRect();
    return rect ? { width: rect.width, height: rect.height } : { width: window.innerWidth, height: window.innerHeight };
  }

  private reposition(): void {
    const rect = this.visibleAnchorRect();
    const frame = this.frame();
    const baseLeft = rect ? rect.left + rect.width / 2 : window.innerWidth / 2;
    const baseTop = rect ? Math.min(rect.bottom, window.innerHeight) - rect.height * BOTTOM_OFFSET_RATIO : window.innerHeight - VIEWPORT_BOTTOM_MARGIN_PX;
    const halfWidth = this.box.offsetWidth / 2;
    const height = this.box.offsetHeight;
    const left = Math.min(Math.max(baseLeft + this.offset.x * frame.width, halfWidth + VIEWPORT_EDGE_MARGIN_PX), window.innerWidth - halfWidth - VIEWPORT_EDGE_MARGIN_PX);
    const top = Math.min(Math.max(baseTop + this.offset.y * frame.height, height + VIEWPORT_EDGE_MARGIN_PX), window.innerHeight - VIEWPORT_EDGE_MARGIN_PX);
    this.box.style.left = `${left}px`;
    this.box.style.top = `${top}px`;
  }
}
