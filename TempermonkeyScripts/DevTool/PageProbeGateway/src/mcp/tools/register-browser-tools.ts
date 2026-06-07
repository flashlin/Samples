import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import {
  debugEvaluateParamsSchema,
  debugGetConsoleParamsSchema,
  debugGetErrorsParamsSchema,
  elementClickParamsSchema,
  elementFillParamsSchema,
  elementGetTextParamsSchema,
  elementTypeParamsSchema,
  networkGetResponseBodyParamsSchema,
  networkListRequestsParamsSchema,
  networkStartCaptureParamsSchema,
  pageHtmlParamsSchema,
  pageMetadataParamsSchema,
  pageNavigateParamsSchema,
  pageReloadParamsSchema,
  pageTextParamsSchema,
  policyListAllowedOriginsParamsSchema,
  snapshotRequestSchema,
  tabGetActiveParamsSchema,
  tabListParamsSchema,
  type BrowserMethod
} from "@page-probe/protocol"
import type { ZodType } from "zod"
import type { BrowserToolService } from "../../application/browser-tool-service"

type BrowserToolDefinition = {
  name: string
  description: string
  method: BrowserMethod
  inputSchema: ZodType
}

const browserToolDefinitions: BrowserToolDefinition[] = [
  {
    name: "browser_list_tabs",
    description: "List tabs available to PageProbe",
    method: "tabs.list",
    inputSchema: tabListParamsSchema
  },
  {
    name: "browser_get_active_tab",
    description: "Get the active browser tab",
    method: "tabs.getActive",
    inputSchema: tabGetActiveParamsSchema
  },
  {
    name: "browser_snapshot",
    description: "Create an accessibility snapshot of a page",
    method: "page.snapshot",
    inputSchema: snapshotRequestSchema
  },
  {
    name: "browser_get_page_metadata",
    description: "Get page metadata",
    method: "page.getMetadata",
    inputSchema: pageMetadataParamsSchema
  },
  {
    name: "browser_get_page_text",
    description: "Get visible page text",
    method: "page.getText",
    inputSchema: pageTextParamsSchema
  },
  {
    name: "browser_get_page_html",
    description: "Get page HTML",
    method: "page.getHtml",
    inputSchema: pageHtmlParamsSchema
  },
  {
    name: "browser_navigate",
    description: "Navigate an existing tab to a URL (target origin must be in the runtime allowlist)",
    method: "page.navigate",
    inputSchema: pageNavigateParamsSchema
  },
  {
    name: "browser_reload",
    description: "Reload an existing tab",
    method: "page.reload",
    inputSchema: pageReloadParamsSchema
  },
  {
    name: "browser_click",
    description: "Click an element from a page snapshot",
    method: "element.click",
    inputSchema: elementClickParamsSchema
  },
  {
    name: "browser_fill",
    description: "Replace an element value",
    method: "element.fill",
    inputSchema: elementFillParamsSchema
  },
  {
    name: "browser_type",
    description: "Type text into an element",
    method: "element.type",
    inputSchema: elementTypeParamsSchema
  },
  {
    name: "browser_get_element_text",
    description: "Get text from an element",
    method: "element.getText",
    inputSchema: elementGetTextParamsSchema
  },
  {
    name: "browser_start_network_capture",
    description: "Start capturing network requests",
    method: "network.startCapture",
    inputSchema: networkStartCaptureParamsSchema
  },
  {
    name: "browser_list_network_requests",
    description: "List captured network requests",
    method: "network.listRequests",
    inputSchema: networkListRequestsParamsSchema
  },
  {
    name: "browser_get_response_body",
    description: "Get a captured network response body",
    method: "network.getResponseBody",
    inputSchema: networkGetResponseBodyParamsSchema
  },
  {
    name: "browser_get_console",
    description: "Get browser console entries",
    method: "debug.getConsole",
    inputSchema: debugGetConsoleParamsSchema
  },
  {
    name: "browser_get_errors",
    description: "Get browser error entries",
    method: "debug.getErrors",
    inputSchema: debugGetErrorsParamsSchema
  },
  {
    name: "browser_evaluate",
    description: "Evaluate JavaScript in a browser tab",
    method: "debug.evaluate",
    inputSchema: debugEvaluateParamsSchema
  },
  {
    name: "browser_list_allowed_origins",
    description: "List origins the LLM is allowed to control (the runtime allowlist)",
    method: "policy.listAllowedOrigins",
    inputSchema: policyListAllowedOriginsParamsSchema
  }
]

export function registerBrowserTools(
  server: McpServer,
  browserToolService: BrowserToolService,
  clientId?: string
): void {
  for (const definition of browserToolDefinitions) {
    server.registerTool(
      definition.name,
      {
        description: definition.description,
        inputSchema: definition.inputSchema
      },
      async (params) => {
        const result = await browserToolService.execute(definition.method, params as never, clientId)
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result)
            }
          ]
        }
      }
    )
  }
}
