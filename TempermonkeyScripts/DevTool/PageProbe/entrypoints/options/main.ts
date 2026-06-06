import { browser } from "wxt/browser";
import {
  ConnectionConfigStore,
  type ConnectionConfig,
} from "../../src/config/connection-config-store";
import {
  CONNECTION_STATE_STORAGE_KEY,
  type ConnectionState,
} from "../../src/connection/connection-state";

const STATUS_LABELS: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  connected: "Connected",
  disconnected: "Disconnected",
};

const connectionConfigStore = new ConnectionConfigStore();
const connectionForm = requireElement<HTMLFormElement>("connection-form");
const gatewayUrlInput = requireElement<HTMLInputElement>("gateway-url");
const tokenInput = requireElement<HTMLInputElement>("token");
const allowedOriginsInput = requireElement<HTMLTextAreaElement>("allowed-origins");
const debugEvaluateEnabledInput = requireElement<HTMLInputElement>(
  "debug-evaluate-enabled",
);
const connectionStatus = requireElement<HTMLOutputElement>("connection-status");
const statusDot = requireElement<HTMLSpanElement>("status-dot");

connectionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void saveAndReconnect();
});

browser.storage.session.onChanged.addListener((changes) => {
  const change = changes[CONNECTION_STATE_STORAGE_KEY];
  if (change) {
    renderConnectionState(change.newValue as ConnectionState | undefined);
  }
});

void initialize();

async function initialize(): Promise<void> {
  try {
    const config = await connectionConfigStore.load();
    gatewayUrlInput.value = config.url;
    tokenInput.value = config.token;
    allowedOriginsInput.value = config.allowedOrigins.join("\n");
    debugEvaluateEnabledInput.checked = config.debugEvaluateEnabled;
    await renderStoredConnectionState();
  } catch (error) {
    connectionStatus.value = formatError("Unable to load settings", error);
  }
}

async function saveAndReconnect(): Promise<void> {
  try {
    await connectionConfigStore.save(readConfig());
    await browser.runtime.sendMessage({ type: "connect" });
  } catch (error) {
    connectionStatus.value = formatError("Unable to save settings", error);
  }
}

async function renderStoredConnectionState(): Promise<void> {
  const stored = await browser.storage.session.get(CONNECTION_STATE_STORAGE_KEY);
  renderConnectionState(stored[CONNECTION_STATE_STORAGE_KEY] as ConnectionState | undefined);
}

function renderConnectionState(state: ConnectionState | undefined): void {
  const resolved: ConnectionState = state ?? "disconnected";
  connectionStatus.value = STATUS_LABELS[resolved];
  statusDot.className = `status-dot ${resolved}`;
}

function readConfig(): ConnectionConfig {
  return {
    url: gatewayUrlInput.value,
    token: tokenInput.value,
    allowedOrigins: allowedOriginsInput.value
      .split("\n")
      .map((origin) => origin.trim())
      .filter(Boolean),
    debugEvaluateEnabled: debugEvaluateEnabledInput.checked,
  };
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);

  if (!(element instanceof HTMLElement)) {
    throw new Error(`Required element not found: ${id}`);
  }

  return element as T;
}

function formatError(message: string, error: unknown): string {
  return error instanceof Error
    ? `${message}: ${error.message}`
    : `${message}: Unknown error`;
}
