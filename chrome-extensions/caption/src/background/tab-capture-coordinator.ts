import { OFFSCREEN_TARGET, type OffscreenCommand, type OffscreenEvent, type TabBroadcast } from "../shared/messages";
import type { StreamServerMessage } from "../shared/stream-protocol";

function tabBroadcastsFor(message: StreamServerMessage): TabBroadcast[] {
  switch (message.type) {
    case "partial":
      return [{ type: "tab-capture-partial", start: message.start, end: message.end, text: message.text }];
    case "covered":
      return [{ type: "tab-capture-segment", start: message.start, end: message.end, display: "" }];
    case "final":
      return [
        { type: "tab-capture-segment", start: message.start, end: message.end, display: "" },
        ...message.segments.map((segment): TabBroadcast => ({ type: "tab-capture-segment", start: segment.start, end: segment.end, display: segment.display })),
      ];
    default:
      return [];
  }
}

const OFFSCREEN_PATH = "offscreen.html";

export class TabCaptureCoordinator {
  private readonly activeTabs = new Set<number>();

  isCapturing(tabId: number): boolean {
    return this.activeTabs.has(tabId);
  }

  async start(tabId: number): Promise<void> {
    if (this.activeTabs.has(tabId)) {
      return;
    }
    await this.ensureOffscreenDocument();
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    this.activeTabs.add(tabId);
    await this.sendToOffscreen({ target: OFFSCREEN_TARGET, type: "offscreen-start", tabId, streamId });
    await this.notifyTab(tabId, { type: "tab-capture-started" });
  }

  async stop(tabId: number): Promise<void> {
    if (!this.activeTabs.has(tabId)) {
      return;
    }
    await this.sendToOffscreen({ target: OFFSCREEN_TARGET, type: "offscreen-stop", tabId });
  }

  async handleOffscreenEvent(event: OffscreenEvent): Promise<void> {
    this.activeTabs.delete(event.tabId);
    await this.notifyTab(event.tabId, { type: "tab-capture-stopped", message: event.message });
    await this.closeOffscreenWhenIdle();
  }

  forwardStreamMessage(tabId: number, message: StreamServerMessage): void {
    for (const broadcast of tabBroadcastsFor(message)) {
      void this.notifyTab(tabId, broadcast);
    }
  }

  private async ensureOffscreenDocument(): Promise<void> {
    const url = chrome.runtime.getURL(OFFSCREEN_PATH);
    const contexts = await chrome.runtime.getContexts({ contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT], documentUrls: [url] });
    if (contexts.length === 0) {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_PATH,
        reasons: [chrome.offscreen.Reason.USER_MEDIA],
        justification: "Capture tab audio for live captions on pages whose media cannot be captured directly",
      });
    }
  }

  private async closeOffscreenWhenIdle(): Promise<void> {
    if (this.activeTabs.size === 0) {
      await chrome.offscreen.closeDocument().catch(() => undefined);
    }
  }

  private async sendToOffscreen(command: OffscreenCommand): Promise<void> {
    await chrome.runtime.sendMessage(command);
  }

  private async notifyTab(tabId: number, message: TabBroadcast): Promise<void> {
    await chrome.tabs.sendMessage(tabId, message, { frameId: 0 }).catch(() => undefined);
  }
}
