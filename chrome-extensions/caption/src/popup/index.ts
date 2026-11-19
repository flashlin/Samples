import { askBackgroundFromPage } from "../shared/extension-page-client";
import type { ExtensionPageRequest, TabStatus } from "../shared/messages";
import { TimelineBar } from "./timeline-bar";

const REFRESH_MILLISECONDS = 1000;

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

interface TabState {
  readonly status: TabStatus;
  readonly tabCapture: boolean;
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing popup element #${id}`);
  }
  return found as T;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const modeText = element<HTMLElement>("mode");
const detailText = element<HTMLElement>("detail");
const serverText = element<HTMLElement>("server");
const recognitionButton = element<HTMLButtonElement>("recognition");
const saveButton = element<HTMLButtonElement>("save");
const captureButton = element<HTMLButtonElement>("capture");
const undoButton = element<HTMLButtonElement>("undo");
const resetButton = element<HTMLButtonElement>("reset");
const optionsButton = element<HTMLButtonElement>("options");

async function activeTabId(): Promise<number> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id === undefined) {
    throw new Error("No active tab");
  }
  return tab.id;
}

function render(state: TabState, timeline: TimelineBar): void {
  const { status, tabCapture } = state;
  modeText.textContent = MODE_LABELS[status.mode];
  detailText.textContent = [status.canonicalUrl, status.message].filter(Boolean).join("\n");
  const recordingMedia = status.recording && !tabCapture;
  recognitionButton.textContent = recordingMedia ? "Stop recognition" : "Start recognition";
  recognitionButton.dataset.recording = String(recordingMedia);
  recognitionButton.disabled = tabCapture;
  saveButton.disabled = !status.canSave;
  captureButton.textContent = tabCapture ? "Stop tab audio capture" : "Capture tab audio";
  captureButton.dataset.capturing = String(tabCapture);
  timeline.render(status.mediaDuration ? { duration: status.mediaDuration, currentTime: status.currentTime, coverage: status.coverage } : null);
}

async function checkServer(): Promise<void> {
  try {
    const health = await askBackgroundFromPage<{ engine: Record<string, string> }>({ type: "check-server" });
    serverText.textContent = `Server OK · ${health.engine.asr ?? "unknown model"}`;
  } catch (error) {
    serverText.textContent = `Server unavailable: ${describe(error)}`;
  }
}

function bindAction(button: HTMLButtonElement, request: () => ExtensionPageRequest, done?: (result: unknown) => string | null): void {
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const result = await askBackgroundFromPage<unknown>(request());
      const message = done?.(result);
      if (message) {
        detailText.textContent = message;
      }
    } catch (error) {
      detailText.textContent = describe(error);
    } finally {
      button.disabled = false;
    }
  });
}

async function main(): Promise<void> {
  const tabId = await activeTabId();
  const timeline = new TimelineBar(element("timeline"), element("gaps"), (time) => void askBackgroundFromPage({ type: "seek", tabId, time }));
  const refresh = async () => render(await askBackgroundFromPage<TabState>({ type: "get-tab-status", tabId }), timeline);
  bindAction(recognitionButton, () => ({ type: recognitionButton.dataset.recording === "true" ? "stop-recognition" : "start-recognition", tabId }));
  bindAction(saveButton, () => ({ type: "save-now", tabId }), () => "Saved");
  bindAction(captureButton, () => ({ type: captureButton.dataset.capturing === "true" ? "stop-tab-capture" : "start-tab-capture", tabId }));
  bindAction(undoButton, () => ({ type: "undo-recording", tabId }), (result) => {
    const outcome = result as { restored: boolean; remainingVersions: number };
    return outcome.restored ? `Restored the version before the last recording (${outcome.remainingVersions} older versions left)` : "Nothing to undo";
  });
  bindAction(resetButton, () => ({ type: "delete-subtitle", tabId }), () => "Saved subtitles for this page were deleted");
  optionsButton.addEventListener("click", () => void chrome.runtime.openOptionsPage());
  await Promise.all([refresh(), checkServer()]);
  setInterval(() => void refresh().catch(() => undefined), REFRESH_MILLISECONDS);
}

void main().catch((error: unknown) => {
  modeText.textContent = describe(error);
});
