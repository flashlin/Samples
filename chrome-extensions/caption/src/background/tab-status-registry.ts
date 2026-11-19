import { IDLE_TAB_STATUS } from "../shared/idle-tab-status";
import type { TabStatus } from "../shared/messages";

const BADGE_BY_MODE: Partial<Record<TabStatus["mode"], { text: string; color: string }>> = {
  playback: { text: "SRT", color: "#2e7d32" },
  recognize: { text: "REC", color: "#c62828" },
  "youtube-captions": { text: "CC", color: "#1565c0" },
  "tab-capture": { text: "TAB", color: "#6a1b9a" },
  error: { text: "!", color: "#ef6c00" },
};

export class TabStatusRegistry {
  private readonly statuses = new Map<number, TabStatus>();

  get(tabId: number): TabStatus {
    return this.statuses.get(tabId) ?? IDLE_TAB_STATUS;
  }

  set(tabId: number, status: TabStatus): void {
    this.statuses.set(tabId, status);
    const badge = BADGE_BY_MODE[status.mode];
    void chrome.action.setBadgeText({ tabId, text: badge?.text ?? "" });
    if (badge) {
      void chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color });
    }
  }

  forget(tabId: number): void {
    this.statuses.delete(tabId);
  }
}
