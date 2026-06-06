import { describe, expect, it } from "bun:test"
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import type { BrowserMethod } from "@page-probe/protocol"
import type { BrowserToolService } from "../src/application/browser-tool-service"
import { registerBrowserTools } from "../src/mcp/tools/register-browser-tools"

type ToolResult = {
  content: Array<{
    type: "text"
    text: string
  }>
}

type ToolHandler = (params: Record<string, unknown>) => Promise<ToolResult>

type ExecuteCall = {
  method: BrowserMethod
  params: unknown
}

class FakeMcpServer {
  readonly handlers = new Map<string, ToolHandler>()

  registerTool(
    name: string,
    _config: unknown,
    handler: ToolHandler
  ): void {
    this.handlers.set(name, handler)
  }
}

class FakeBrowserToolService {
  readonly calls: ExecuteCall[] = []

  constructor(private readonly result: unknown) {}

  async execute(method: BrowserMethod, params: unknown): Promise<unknown> {
    this.calls.push({ method, params })
    return this.result
  }
}

describe("registerBrowserTools", () => {
  it.each([
    {
      tool: "browser_list_tabs",
      method: "tabs.list",
      params: {},
      result: {
        tabs: [
          {
            id: 7,
            active: true,
            title: "PageProbe",
            url: "https://example.com"
          }
        ]
      }
    },
    {
      tool: "browser_snapshot",
      method: "page.snapshot",
      params: {
        tabId: 7,
        interactiveOnly: true,
        maxDepth: 8
      },
      result: {
        snapshotId: "snapshot-1",
        tabId: 7,
        documentEpoch: 1,
        origin: "https://example.com",
        text: "button Submit",
        refs: {},
        truncated: false
      }
    },
    {
      tool: "browser_click",
      method: "element.click",
      params: {
        tabId: 7,
        snapshotId: "snapshot-1",
        ref: "@e1"
      },
      result: {
        success: true
      }
    },
    {
      tool: "browser_evaluate",
      method: "debug.evaluate",
      params: {
        tabId: 7,
        expression: "document.title",
        returnByValue: true
      },
      result: {
        value: "PageProbe",
        type: "string"
      }
    }
  ] as const)("maps $tool to $method and returns JSON text", async ({
    tool,
    method,
    params,
    result
  }) => {
    const server = new FakeMcpServer()
    const service = new FakeBrowserToolService(result)

    registerBrowserTools(
      server as unknown as McpServer,
      service as unknown as BrowserToolService
    )

    const handler = server.handlers.get(tool)

    expect(handler).toBeDefined()
    expect(await handler!(params)).toEqual({
      content: [
        {
          type: "text",
          text: JSON.stringify(result)
        }
      ]
    })
    expect(service.calls).toEqual([
      {
        method,
        params
      }
    ])
  })

  it("registers every MVP browser tool", () => {
    const server = new FakeMcpServer()
    const service = new FakeBrowserToolService({})

    registerBrowserTools(
      server as unknown as McpServer,
      service as unknown as BrowserToolService
    )

    expect([...server.handlers.keys()]).toEqual([
      "browser_list_tabs",
      "browser_get_active_tab",
      "browser_snapshot",
      "browser_get_page_metadata",
      "browser_get_page_text",
      "browser_get_page_html",
      "browser_click",
      "browser_fill",
      "browser_type",
      "browser_get_element_text",
      "browser_start_network_capture",
      "browser_list_network_requests",
      "browser_get_response_body",
      "browser_get_console",
      "browser_get_errors",
      "browser_evaluate",
      "browser_list_allowed_origins"
    ])
  })
})
