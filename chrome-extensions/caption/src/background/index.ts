import {
  contentRequestSchema,
  errorReply,
  extensionPageRequestSchema,
  offscreenEventSchema,
  okReply,
  type ContentRequest,
  type ExtensionPageRequest,
  type Reply,
  type TabBroadcast,
} from "../shared/messages";
import { CaptionServerClient } from "./caption-server-client";
import { loadConfig, saveOverlayOffset } from "./config-repository";
import { identifyPage, resolvePage } from "./page-resolver";
import { relayPort, type RelayConnection } from "./stream-relay";
import { TabCaptureCoordinator } from "./tab-capture-coordinator";
import { CONTENT_STREAM_PORT, TAB_STREAM_PORT_PREFIX } from "../shared/stream-protocol";
import { TabStatusRegistry } from "./tab-status-registry";

const OFFSCREEN_URL = chrome.runtime.getURL("offscreen.html");
const EXTENSION_ORIGIN = chrome.runtime.getURL("");

const statuses = new TabStatusRegistry();

async function createClient(): Promise<CaptionServerClient> {
  const config = await loadConfig();
  return new CaptionServerClient({ endpoint: config.server.endpoint, token: config.server.token, timeoutSeconds: config.server.requestTimeoutSeconds });
}

const tabCapture = new TabCaptureCoordinator();

async function relayConnection(): Promise<RelayConnection> {
  const config = await loadConfig();
  if (!config.server.token) {
    throw new Error("Caption server token is not configured, open the extension options");
  }
  return { endpoint: config.server.endpoint, token: config.server.token };
}

chrome.runtime.onConnect.addListener((port) => {
  const sender = port.sender;
  if (sender?.id !== chrome.runtime.id) {
    port.disconnect();
    return;
  }
  if (port.name === CONTENT_STREAM_PORT && sender.tab?.id !== undefined) {
    relayPort(port, relayConnection(), (message) => {
      try {
        port.postMessage(message);
      } catch {
        return;
      }
    });
    return;
  }
  const tabId = Number(port.name.slice(TAB_STREAM_PORT_PREFIX.length));
  if (port.name.startsWith(TAB_STREAM_PORT_PREFIX) && sender.url === OFFSCREEN_URL && Number.isInteger(tabId)) {
    relayPort(port, relayConnection(), (message) => tabCapture.forwardStreamMessage(tabId, message));
    return;
  }
  port.disconnect();
});

async function handleContentRequest(request: ContentRequest, tabId: number): Promise<unknown> {
  switch (request.type) {
    case "resolve-page":
      return resolvePage(request.pageUrl, await loadConfig(), await createClient());
    case "translate-lines":
      return (await createClient()).translateLines(request.texts);
    case "save-subtitle":
      return (await createClient()).putSubtitle(request.urlKey, {
        meta: request.meta,
        cues: request.cues,
        placement: request.placement,
        expectedRevision: request.expectedRevision,
        confirmOverwrite: request.confirmOverwrite,
      });
    case "save-overlay-offset":
      return saveOverlayOffset(request.x, request.y);
    case "report-status":
      statuses.set(tabId, request.status);
      return null;
  }
}

async function urlKeyOfTab(tabId: number): Promise<string> {
  const tab = await chrome.tabs.get(tabId);
  const identity = tab.url ? await identifyPage(tab.url, await loadConfig()) : null;
  if (identity === null) {
    throw new Error("This tab has no subtitle key");
  }
  return identity.urlKey;
}

async function tellTab(tabId: number, message: TabBroadcast): Promise<void> {
  await chrome.tabs.sendMessage(tabId, message, { frameId: 0 }).catch(() => undefined);
}

async function deleteSubtitleForTab(tabId: number): Promise<void> {
  await (await createClient()).deleteSubtitle(await urlKeyOfTab(tabId));
  await tellTab(tabId, { type: "subtitle-changed" });
}

async function undoRecordingForTab(tabId: number): Promise<{ restored: boolean; remainingVersions: number }> {
  const urlKey = await urlKeyOfTab(tabId);
  await tellTab(tabId, { type: "recognition-stop", discard: true });
  const outcome = await (await createClient()).undoSubtitle(urlKey);
  await tellTab(tabId, { type: "subtitle-changed" });
  return outcome;
}

async function handleExtensionPageRequest(request: ExtensionPageRequest): Promise<unknown> {
  switch (request.type) {
    case "get-tab-status":
      return { status: statuses.get(request.tabId), tabCapture: tabCapture.isCapturing(request.tabId) };
    case "delete-subtitle":
      return deleteSubtitleForTab(request.tabId);
    case "undo-recording":
      return undoRecordingForTab(request.tabId);
    case "start-recognition":
      return tellTab(request.tabId, { type: "recognition-start" });
    case "stop-recognition":
      return tellTab(request.tabId, { type: "recognition-stop", discard: false });
    case "save-now":
      return tellTab(request.tabId, { type: "save-now" });
    case "seek":
      return tellTab(request.tabId, { type: "seek", time: request.time });
    case "start-tab-capture":
      return tabCapture.start(request.tabId);
    case "stop-tab-capture":
      return tabCapture.stop(request.tabId);
    case "check-server":
      return (await createClient()).health();
  }
}

function route(message: unknown, sender: chrome.runtime.MessageSender): Promise<unknown> | null {
  if (sender.id !== chrome.runtime.id) {
    return null;
  }
  if (sender.url === OFFSCREEN_URL) {
    const event = offscreenEventSchema.safeParse(message);
    return event.success ? tabCapture.handleOffscreenEvent(event.data) : null;
  }
  if (sender.url?.startsWith(EXTENSION_ORIGIN)) {
    return handleExtensionPageRequest(extensionPageRequestSchema.parse(message));
  }
  if (sender.tab?.id !== undefined) {
    return handleContentRequest(contentRequestSchema.parse(message), sender.tab.id);
  }
  return null;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse: (reply: Reply<unknown>) => void) => {
  let pending: Promise<unknown> | null;
  try {
    pending = route(message, sender);
  } catch (error) {
    sendResponse(errorReply(error));
    return false;
  }
  if (pending === null) {
    return false;
  }
  pending.then((value) => sendResponse(okReply(value ?? null))).catch((error: unknown) => sendResponse(errorReply(error)));
  return true;
});

chrome.tabs.onRemoved.addListener((tabId) => {
  statuses.forget(tabId);
  void tabCapture.stop(tabId);
});
