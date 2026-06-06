import { describe, expect, test } from "bun:test";
import {
  ConnectionConfigStore,
  type ConnectionConfig,
} from "../src/config/connection-config-store";

class MockStorageArea {
  readonly values: Record<string, unknown> = {};

  async get(key: string): Promise<Record<string, unknown>> {
    return { [key]: this.values[key] };
  }

  async set(items: Record<string, unknown>): Promise<void> {
    Object.assign(this.values, items);
  }
}

describe("ConnectionConfigStore", () => {
  test("loads the default configuration when storage is empty", async () => {
    const store = new ConnectionConfigStore(new MockStorageArea());

    const config = await store.load();

    expect(config).toEqual({
      url: "ws://127.0.0.1:17890/extension",
      token: "",
      allowedOrigins: [],
      debugEvaluateEnabled: false,
    });
  });

  test("saves and loads a normalized configuration", async () => {
    const storage = new MockStorageArea();
    const store = new ConnectionConfigStore(storage);
    const config: ConnectionConfig = {
      url: "  wss://gateway.example.com/extension  ",
      token: "  secret-token  ",
      allowedOrigins: [
        "https://example.com/",
        "https://example.com",
        " http://localhost:3000 ",
      ],
      debugEvaluateEnabled: true,
    };

    await store.save(config);

    expect(await store.load()).toEqual({
      url: "wss://gateway.example.com/extension",
      token: "secret-token",
      allowedOrigins: [
        "https://example.com",
        "http://localhost:3000",
      ],
      debugEvaluateEnabled: true,
    });
  });

  test("loads defaults when stored configuration is invalid", async () => {
    const storage = new MockStorageArea();
    storage.values.connectionConfig = {
      url: "file:///tmp/gateway",
      token: "secret-token",
      allowedOrigins: ["https://example.com"],
      debugEvaluateEnabled: false,
    };
    const store = new ConnectionConfigStore(storage);

    expect(await store.load()).toEqual({
      url: "ws://127.0.0.1:17890/extension",
      token: "",
      allowedOrigins: [],
      debugEvaluateEnabled: false,
    });
  });

  test("rejects an unsupported Gateway URL", async () => {
    const store = new ConnectionConfigStore(new MockStorageArea());

    await expect(
      store.save({
        url: "file:///tmp/gateway",
        token: "secret-token",
        allowedOrigins: ["https://example.com"],
        debugEvaluateEnabled: false,
      }),
    ).rejects.toThrow("Gateway URL must be a valid WebSocket URL.");
  });

  test("rejects an empty token", async () => {
    const store = new ConnectionConfigStore(new MockStorageArea());

    await expect(
      store.save({
        url: "ws://127.0.0.1:17890/extension",
        token: " ",
        allowedOrigins: ["https://example.com"],
        debugEvaluateEnabled: false,
      }),
    ).rejects.toThrow("Token is required.");
  });

  test("rejects an empty runtime allowlist", async () => {
    const store = new ConnectionConfigStore(new MockStorageArea());

    await expect(
      store.save({
        url: "ws://127.0.0.1:17890/extension",
        token: "secret-token",
        allowedOrigins: [],
        debugEvaluateEnabled: false,
      }),
    ).rejects.toThrow("At least one allowed origin is required.");
  });

  test("rejects an invalid allowed origin", async () => {
    const store = new ConnectionConfigStore(new MockStorageArea());

    await expect(
      store.save({
        url: "ws://127.0.0.1:17890/extension",
        token: "secret-token",
        allowedOrigins: ["https://example.com/path"],
        debugEvaluateEnabled: false,
      }),
    ).rejects.toThrow("Invalid allowed origin");
  });
});
