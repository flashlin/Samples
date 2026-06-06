import { BrowserToolService } from "./application/browser-tool-service"
import { BearerAuth } from "./auth/bearer-auth"
import { RequestOriginGuard } from "./auth/request-origin-guard"
import { CommandDispatcher } from "./command-bus/command-dispatcher"
import { PendingRequestStore } from "./command-bus/pending-request-store"
import { TimeoutPolicy } from "./command-bus/timeout-policy"
import { loadServerConfig } from "./config/server-config"
import { TokenStore } from "./config/token-store"
import { ConnectionRegistry } from "./connections/connection-registry"
import { startHttpServer } from "./http/http-server"
import { createClientsRoute } from "./http/routes/clients-route"
import { createHealthRoute } from "./http/routes/health-route"
import { createMcpServer } from "./mcp/mcp-server"
import { McpSessionStore } from "./mcp/mcp-session-store"

const config = loadServerConfig()
const tokenStore = new TokenStore(config.tokenPath)
const token = await tokenStore.load(process.env.DEVTOOL_TOKEN)
const auth = new BearerAuth(token)
const originGuard = new RequestOriginGuard(config.allowedOrigins, config.allowedHosts)
const connectionRegistry = new ConnectionRegistry()
const pendingRequestStore = new PendingRequestStore()
const timeoutPolicy = new TimeoutPolicy(config.requestTimeoutMs)
const dispatcher = new CommandDispatcher(connectionRegistry, pendingRequestStore, timeoutPolicy)
const browserToolService = new BrowserToolService(dispatcher)
const routeSecurity = {
  bearerAuth: auth,
  requestOriginGuard: originGuard
}
const mcpSessionStore = new McpSessionStore((clientId) => createMcpServer(browserToolService, clientId))

const server = startHttpServer({
  config,
  auth,
  originGuard,
  connectionRegistry,
  pendingRequestStore,
  browserToolService,
  mcpHandler: (request) => {
    const clientId = request.headers.get("x-page-probe-client") ?? undefined
    return mcpSessionStore.handle(request, clientId)
  },
  healthRoute: createHealthRoute(routeSecurity),
  clientsRoute: createClientsRoute({
    ...routeSecurity,
    connectionRegistry
  })
})

console.log(`PageProbeGateway listening on http://${server.hostname}:${server.port}`)
