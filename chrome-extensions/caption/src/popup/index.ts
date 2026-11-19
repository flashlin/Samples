import { askBackgroundFromPage } from "../shared/extension-page-client";
import type { TabStatus } from "../shared/messages";

const MODE_LABELS: Record<TabStatus["mode"], string> = {
  idle: "Waiting for media to play",
  disabled: "Disabled on this site",
  playback: "Showing saved subtitles",
  recognize: "Recognizing speech",
  "youtube-captions": "Using YouTube captions",
  "tab-capture": "Capturing tab audio",
  closed: "Closed on this page",
  error: "Error",
};

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing popup element #${id}`);
  }
  return found as T;
}

const modeText = element<HTMLElement>("mode");
const detailText = element<HTMLElement>("detail");
const serverText = element<HTMLElement>("server");
const captureButton = element<HTMLButtonElement>("capture");
const resetButton = element<HTMLButtonElement>("reset");
const optionsButton = element<HTMLButtonElement>("options");

async function activeTabId(): Promise<number> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id === undefined) {
    throw new Error("No active tab");
  }
  return tab.id;
}

async function refresh(tabId: number): Promise<void> {
  const { status, tabCapture } = await askBackgroundFromPage<{ status: TabStatus; tabCapture: boolean }>({ type: "get-tab-status", tabId });
  modeText.textContent = MODE_LABELS[status.mode];
  const details = [
    status.canonicalUrl,
    status.coveredSeconds > 0 ? `Covered ${Math.round(status.coveredSeconds)} s, ${status.cueCount} cues` : null,
    status.message,
  ].filter(Boolean);
  detailText.textContent = details.join("\n");
  captureButton.textContent = tabCapture ? "Stop tab audio capture" : "Capture tab audio";
  captureButton.dataset.capturing = String(tabCapture);
}

async function checkServer(): Promise<void> {
  try {
    const health = await askBackgroundFromPage<{ engine: Record<string, string> }>({ type: "check-server" });
    serverText.textContent = `Server OK · ${health.engine.asr ?? "unknown model"}`;
  } catch (error) {
    serverText.textContent = `Server unavailable: ${error instanceof Error ? error.message : String(error)}`;
  }
}

async function main(): Promise<void> {
  const tabId = await activeTabId();
  captureButton.addEventListener("click", async () => {
    const capturing = captureButton.dataset.capturing === "true";
    try {
      await askBackgroundFromPage({ type: capturing ? "stop-tab-capture" : "start-tab-capture", tabId });
    } catch (error) {
      detailText.textContent = error instanceof Error ? error.message : String(error);
    }
    await refresh(tabId);
  });
  resetButton.addEventListener("click", async () => {
    try {
      await askBackgroundFromPage({ type: "delete-subtitle", tabId });
      detailText.textContent = "Saved subtitles for this page were deleted";
    } catch (error) {
      detailText.textContent = error instanceof Error ? error.message : String(error);
    }
  });
  optionsButton.addEventListener("click", () => void chrome.runtime.openOptionsPage());
  await Promise.all([refresh(tabId), checkServer()]);
}

void main().catch((error: unknown) => {
  modeText.textContent = error instanceof Error ? error.message : String(error);
});
