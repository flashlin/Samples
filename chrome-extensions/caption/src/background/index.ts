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
import { TabCaptureCoordinator } from "./tab-capture-coordinator";
import { TabStatusRegistry } from "./tab-status-registry";

const OFFSCREEN_URL = chrome.runtime.getURL("offscreen.html");
const EXTENSION_ORIGIN = chrome.runtime.getURL("");

const statuses = new TabStatusRegistry();

async function createClient(): Promise<CaptionServerClient> {
  const config = await loadConfig();
  return new CaptionServerClient({ endpoint: config.server.endpoint, token: config.server.token, timeoutSeconds: config.server.requestTimeoutSeconds });
}

const tabCapture = new TabCaptureCoordinator(createClient, async () => (await loadConfig()).chunk);

async function handleContentRequest(request: ContentRequest, tabId: number): Promise<unknown> {
  switch (request.type) {
    case "resolve-page":
      return resolvePage(request.pageUrl, await loadConfig(), await createClient());
    case "transcribe":
      return (await createClient()).transcribe(request.audioBase64, request.mimeType, !request.interim);
    case "translate-lines":
      return (await createClient()).translateLines(request.texts);
    case "save-subtitle":
      return (await createClient()).putSubtitle(request.urlKey, request.meta, request.srt, request.expectedRevision);
    case "save-overlay-offset":
      return saveOverlayOffset(request.x, request.y);
    case "report-status":
      statuses.set(tabId, request.status);
      return null;
  }
}

async function deleteSubtitleForTab(tabId: number): Promise<void> {
  const tab = await chrome.tabs.get(tabId);
  const identity = tab.url ? await identifyPage(tab.url, await loadConfig()) : null;
  if (identity === null) {
    throw new Error("This tab has no subtitle key");
  }
  await (await createClient()).deleteSubtitle(identity.urlKey);
  const message: TabBroadcast = { type: "subtitle-deleted" };
  await chrome.tabs.sendMessage(tabId, message).catch(() => undefined);
}

async function handleExtensionPageRequest(request: ExtensionPageRequest): Promise<unknown> {
  switch (request.type) {
    case "get-tab-status":
      return { status: statuses.get(request.tabId), tabCapture: tabCapture.isCapturing(request.tabId) };
    case "delete-subtitle":
      return deleteSubtitleForTab(request.tabId);
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
