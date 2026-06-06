import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import type { BrowserToolService } from "../application/browser-tool-service"
import { registerBrowserTools } from "./tools/register-browser-tools"

export function createMcpServer(
  browserToolService: BrowserToolService,
  clientId?: string
): McpServer {
  const server = new McpServer({
    name: "page-probe",
    version: "0.1.0"
  })
  registerBrowserTools(server, browserToolService, clientId)
  return server
}
