import { browser } from "wxt/browser";

export interface ConnectionConfig {
  url: string;
  token: string;
  allowedOrigins: string[];
  debugEvaluateEnabled: boolean;
}

interface StorageArea {
  get(key: string): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
}

const STORAGE_KEY = "connectionConfig";
const DEFAULT_CONNECTION_CONFIG: ConnectionConfig = {
  url: "ws://127.0.0.1:17890/extension",
  token: readInjectedToken(),
  allowedOrigins: [],
  debugEvaluateEnabled: false,
};
const SUPPORTED_PROTOCOLS = new Set(["ws:", "wss:"]);

export class ConnectionConfigStore {
  constructor(private readonly storage: StorageArea = browser.storage.local) {}

  async load(): Promise<ConnectionConfig> {
    const storedValues = await this.storage.get(STORAGE_KEY);
    const storedConfig = storedValues[STORAGE_KEY];

    if (!isConnectionConfig(storedConfig)) {
      return { ...DEFAULT_CONNECTION_CONFIG };
    }

    return storedConfig;
  }

  async save(config: ConnectionConfig): Promise<void> {
    const normalizedConfig = normalizeConnectionConfig(config);
    validateConnectionConfig(normalizedConfig);
    await this.storage.set({ [STORAGE_KEY]: normalizedConfig });
  }
}

function normalizeConnectionConfig(config: ConnectionConfig): ConnectionConfig {
  return {
    url: config.url.trim(),
    token: config.token.trim(),
    allowedOrigins: [...new Set(config.allowedOrigins.map(normalizeOrigin))],
    debugEvaluateEnabled: config.debugEvaluateEnabled,
  };
}

function validateConnectionConfig(config: ConnectionConfig): void {
  if (!isSupportedGatewayUrl(config.url)) {
    throw new Error("Gateway URL must be a valid WebSocket URL.");
  }

  if (config.token.length === 0) {
    throw new Error("Token is required.");
  }

  if (config.allowedOrigins.length === 0) {
    throw new Error("At least one allowed origin is required.");
  }
}

function isConnectionConfig(value: unknown): value is ConnectionConfig {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.url === "string" &&
    typeof value.token === "string" &&
    Array.isArray(value.allowedOrigins) &&
    value.allowedOrigins.every((origin) => typeof origin === "string" && isSupportedOrigin(origin)) &&
    typeof value.debugEvaluateEnabled === "boolean" &&
    isSupportedGatewayUrl(value.url)
  );
}

function normalizeOrigin(value: string): string {
  const trimmed = value.trim();
  try {
    const url = new URL(trimmed);
    if (
      (url.protocol !== "http:" && url.protocol !== "https:") ||
      (url.pathname !== "" && url.pathname !== "/") ||
      url.search ||
      url.hash
    ) {
      throw new Error();
    }
    return url.origin;
  } catch {
    throw new Error(`Invalid allowed origin: ${trimmed}`);
  }
}

function isSupportedOrigin(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.origin === value &&
      url.pathname === "/" &&
      !url.search &&
      !url.hash
    );
  } catch {
    return false;
  }
}

function isSupportedGatewayUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return SUPPORTED_PROTOCOLS.has(url.protocol) && url.hostname.length > 0;
  } catch {
    return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function readInjectedToken(): string {
  return import.meta.env.WXT_GATEWAY_TOKEN ?? "";
}
