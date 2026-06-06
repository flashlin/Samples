import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js"
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js"
import { BrowserProtocolError } from "@page-probe/protocol"

type McpSession = {
  server: McpServer
  transport: WebStandardStreamableHTTPServerTransport
  lastAccessedAt: number
}

type McpServerFactory = (clientId?: string) => McpServer

export class McpSessionStore {
  readonly sessions = new Map<string, McpSession>()

  constructor(
    private readonly createServer: McpServerFactory,
    private readonly maxSessions = 16,
    private readonly idleTimeoutMs = 30 * 60 * 1000
  ) {}

  async handle(request: Request, clientId?: string): Promise<Response> {
    await this.removeExpired()
    const sessionId = request.headers.get("mcp-session-id")
    if (sessionId) {
      const session = this.sessions.get(sessionId)
      if (!session) {
        return mcpErrorResponse(404, "MCP session was not found")
      }
      session.lastAccessedAt = Date.now()
      const response = await session.transport.handleRequest(request)
      if (request.method === "DELETE") {
        await this.close(sessionId)
      }
      return response
    }

    if (request.method !== "POST") {
      return mcpErrorResponse(400, "MCP session ID is required")
    }

    const parsedBody = await request.json()
    if (!isInitializeRequest(parsedBody)) {
      return mcpErrorResponse(400, "MCP initialize request is required")
    }
    if (this.sessions.size >= this.maxSessions) {
      return mcpErrorResponse(429, "MCP session limit reached")
    }

    return this.initialize(request, parsedBody, clientId)
  }

  async close(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId)
    if (!session) {
      return
    }
    this.sessions.delete(sessionId)
    await session.transport.close()
    await session.server.close()
  }

  private async initialize(request: Request, parsedBody: unknown, clientId?: string): Promise<Response> {
    const server = this.createServer(clientId)
    let transport: WebStandardStreamableHTTPServerTransport
    transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),
      onsessioninitialized: (sessionId) => {
        this.sessions.set(sessionId, {
          server,
          transport,
          lastAccessedAt: Date.now()
        })
      }
    })
    transport.onclose = () => {
      if (transport.sessionId) {
        this.sessions.delete(transport.sessionId)
      }
    }
    await server.connect(transport)

    try {
      return await transport.handleRequest(request, { parsedBody })
    } catch (error) {
      await transport.close()
      await server.close()
      throw new BrowserProtocolError(
        "INTERNAL_ERROR",
        error instanceof Error ? error.message : "MCP initialization failed"
      )
    }
  }

  private async removeExpired(): Promise<void> {
    const threshold = Date.now() - this.idleTimeoutMs
    const expired = [...this.sessions.entries()]
      .filter(([, session]) => session.lastAccessedAt < threshold)
      .map(([sessionId]) => sessionId)
    await Promise.all(expired.map((sessionId) => this.close(sessionId)))
  }
}

function mcpErrorResponse(status: number, message: string): Response {
  return Response.json({
    jsonrpc: "2.0",
    error: {
      code: -32_000,
      message
    },
    id: null
  }, { status })
}
