import { describe, expect, it } from "bun:test"
import type { BrowserMethod } from "@page-probe/protocol"
import type { BrowserToolService } from "../src/application/browser-tool-service"
import { createMcpServer } from "../src/mcp/mcp-server"
import { McpSessionStore } from "../src/mcp/mcp-session-store"

class FakeBrowserToolService {
  async execute(method: BrowserMethod): Promise<unknown> {
    return method === "tabs.list" ? { tabs: [] } : {}
  }
}

describe("MCP session lifecycle", () => {
  it("initializes, lists tools, and closes a stateful session", async () => {
    const service = new FakeBrowserToolService() as unknown as BrowserToolService
    const store = new McpSessionStore((clientId) => createMcpServer(service, clientId))
    const initializeResponse = await store.handle(createMcpRequest({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-03-26",
        capabilities: {},
        clientInfo: {
          name: "integration-test",
          version: "1.0.0"
        }
      }
    }), "chrome-test")

    expect(initializeResponse.status).toBe(200)
    const sessionId = initializeResponse.headers.get("mcp-session-id")
    expect(sessionId).toBeTruthy()
    expect(await readSseData(initializeResponse)).toMatchObject({
      result: {
        serverInfo: {
          name: "page-probe",
          version: "0.1.0"
        }
      },
      id: 1
    })

    const initializedResponse = await store.handle(createMcpRequest({
      jsonrpc: "2.0",
      method: "notifications/initialized"
    }, sessionId!))
    expect(initializedResponse.status).toBe(202)

    const toolsResponse = await store.handle(createMcpRequest({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {}
    }, sessionId!))
    const toolsMessage = await readSseData(toolsResponse)
    const tools = readTools(toolsMessage)
    expect(tools).toHaveLength(21)
    expect(tools.map(({ name }) => name)).toContain("browser_snapshot")
    expect(tools.map(({ name }) => name)).toContain("browser_evaluate")
    expect(tools.map(({ name }) => name)).toContain("browser_list_allowed_origins")
    expect(tools.map(({ name }) => name)).toContain("browser_navigate")
    expect(tools.map(({ name }) => name)).toContain("browser_reload")
    expect(tools.map(({ name }) => name)).toContain("browser_go_back")
    expect(tools.map(({ name }) => name)).toContain("browser_go_forward")

    const deleteResponse = await store.handle(new Request("http://127.0.0.1:17890/mcp", {
      method: "DELETE",
      headers: {
        "mcp-session-id": sessionId!,
        "mcp-protocol-version": "2025-03-26"
      }
    }))
    expect(deleteResponse.status).toBe(200)
    expect(store.sessions.size).toBe(0)
  })

  it("rejects an unknown session", async () => {
    const service = new FakeBrowserToolService() as unknown as BrowserToolService
    const store = new McpSessionStore(() => createMcpServer(service))
    const response = await store.handle(createMcpRequest({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {}
    }, "missing-session"))

    expect(response.status).toBe(404)
  })
})

function createMcpRequest(body: unknown, sessionId?: string): Request {
  const headers = new Headers({
    accept: "application/json, text/event-stream",
    "content-type": "application/json",
    "mcp-protocol-version": "2025-03-26"
  })
  if (sessionId) {
    headers.set("mcp-session-id", sessionId)
  }
  return new Request("http://127.0.0.1:17890/mcp", {
    method: "POST",
    headers,
    body: JSON.stringify(body)
  })
}

async function readSseData(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text()
  const data = text.split("\n").find((line) => line.startsWith("data: "))
  if (!data) {
    throw new Error("SSE data was not found")
  }
  return JSON.parse(data.slice("data: ".length)) as Record<string, unknown>
}

function readTools(message: Record<string, unknown>): Array<{ name: string }> {
  const result = message.result
  if (!isRecord(result) || !Array.isArray(result.tools)) {
    throw new Error("MCP tools were not found")
  }
  return result.tools.filter(isNamedTool)
}

function isNamedTool(value: unknown): value is { name: string } {
  return isRecord(value) && typeof value.name === "string"
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}
