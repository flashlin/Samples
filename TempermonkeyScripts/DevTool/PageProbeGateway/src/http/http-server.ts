import type { Server, ServerWebSocket } from "bun"
import {
  authMessageSchema,
  authResultSchema,
  heartbeatSchema,
  protocolVersion
} from "@page-probe/protocol"
import type { BrowserToolService } from "../application/browser-tool-service"
import type { BearerAuth } from "../auth/bearer-auth"
import type { RequestOriginGuard } from "../auth/request-origin-guard"
import type { PendingRequestStore } from "../command-bus/pending-request-store"
import type { ConnectionRegistry } from "../connections/connection-registry"
import { ExtensionConnection, type ExtensionSocketData } from "../connections/extension-connection"
import type { ServerConfig } from "../config/server-config"
import { handleBrowserRoute } from "./browser-routes"
import { mapErrorToResponse } from "./error-mapper"

type ServerDependencies = {
  config: ServerConfig
  auth: BearerAuth
  originGuard: RequestOriginGuard
  connectionRegistry: ConnectionRegistry
  pendingRequestStore: PendingRequestStore
  browserToolService: BrowserToolService
  mcpHandler?: (request: Request) => Promise<Response>
  healthRoute?: (request: Request) => Response
  clientsRoute?: (request: Request) => Response
}

export function startHttpServer(dependencies: ServerDependencies): Server<ExtensionSocketData> {
  return Bun.serve<ExtensionSocketData>({
    hostname: dependencies.config.hostname,
    port: dependencies.config.port,
    maxRequestBodySize: 1_048_576,
    fetch: (request, server) => handleRequest(request, server, dependencies),
    websocket: {
      open(socket) {
        setTimeout(() => {
          if (!socket.data.authenticated) {
            socket.close(4003, "Authentication timeout")
          }
        }, 5_000)
        startKeepAlive(socket, dependencies.config.keepAliveIntervalMs)
      },
      message(socket, message) {
        handleSocketMessage(socket, message, dependencies)
      },
      close(socket) {
        stopKeepAlive(socket)
        const clientId = socket.data.clientId
        if (clientId) {
          dependencies.connectionRegistry.remove(clientId)
          dependencies.pendingRequestStore.rejectClient(clientId)
        }
      }
    }
  })
}

async function handleRequest(
  request: Request,
  server: Server<ExtensionSocketData>,
  dependencies: ServerDependencies
): Promise<Response> {
  try {
    const url = new URL(request.url)

    if (url.pathname === "/extension") {
      dependencies.originGuard.assertExtensionRequest(request)
      const upgraded = server.upgrade(request, {
        data: {
          authenticated: false,
          connectedAt: Date.now(),
          lastPongAt: Date.now()
        }
      })
      return upgraded ? new Response(null, { status: 101 }) : new Response("WebSocket upgrade failed", { status: 400 })
    }

    if (request.method === "GET" && url.pathname === "/health" && dependencies.healthRoute) {
      return dependencies.healthRoute(request)
    }
    if (request.method === "GET" && url.pathname === "/clients" && dependencies.clientsRoute) {
      return dependencies.clientsRoute(request)
    }

    dependencies.originGuard.assertRequest(request)
    dependencies.auth.assertRequest(request)
    if (url.pathname === "/mcp" && dependencies.mcpHandler) {
      return dependencies.mcpHandler(request)
    }

    const browserResponse = await handleBrowserRoute(request, url.pathname, dependencies.browserToolService)
    return browserResponse ?? Response.json({ error: { code: "NOT_FOUND", message: "Route not found" } }, { status: 404 })
  } catch (error) {
    return mapErrorToResponse(error)
  }
}

function startKeepAlive(
  socket: ServerWebSocket<ExtensionSocketData>,
  intervalMs: number
): void {
  socket.data.keepAliveTimer = setInterval(() => {
    const sent = socket.send(JSON.stringify({
      protocolVersion,
      type: "ping",
      timestamp: Date.now()
    }))
    if (sent < 0) {
      stopKeepAlive(socket)
    }
  }, intervalMs)
}

function stopKeepAlive(socket: ServerWebSocket<ExtensionSocketData>): void {
  if (socket.data.keepAliveTimer) {
    clearInterval(socket.data.keepAliveTimer)
    socket.data.keepAliveTimer = undefined
  }
}

function handleSocketMessage(
  socket: ServerWebSocket<ExtensionSocketData>,
  message: string | Buffer,
  dependencies: ServerDependencies
): void {
  try {
    const input = JSON.parse(message.toString()) as unknown
    if (!socket.data.authenticated) {
      authenticateSocket(socket, input, dependencies)
      return
    }

    const heartbeat = heartbeatSchema.safeParse(input)
    if (heartbeat.success) {
      if (heartbeat.data.type === "ping") {
        socket.send(JSON.stringify({ ...heartbeat.data, type: "pong" }))
      } else {
        socket.data.lastPongAt = Date.now()
      }
      return
    }

    dependencies.pendingRequestStore.settleUnknown(input)
  } catch {
    socket.close(4002, "Invalid protocol message")
  }
}

function authenticateSocket(
  socket: ServerWebSocket<ExtensionSocketData>,
  input: unknown,
  dependencies: ServerDependencies
): void {
  const authMessage = authMessageSchema.parse(input)
  if (!dependencies.auth.matches(authMessage.token)) {
    socket.send(JSON.stringify(authResultSchema.parse({
      protocolVersion,
      type: "authResult",
      ok: false,
      error: { code: "PERMISSION_DENIED", message: "Authentication failed" }
    })))
    socket.close(4003, "Authentication failed")
    return
  }

  socket.data.authenticated = true
  socket.data.clientId = authMessage.client.clientId
  dependencies.connectionRegistry.register(new ExtensionConnection(
    authMessage.client.clientId,
    authMessage.client.name,
    authMessage.client.version,
    socket
  ))
  socket.send(JSON.stringify({
    protocolVersion,
    type: "authResult",
    ok: true
  }))
}
