import { CdpClient } from "../src/cdp/cdp-client"
import { TabSessionManager } from "../src/cdp/tab-session-manager"
import { CommandProcessor } from "../src/commands/command-processor"
import { HandlerRegistry } from "../src/commands/handler-registry"
import { registerHandlers } from "../src/commands/register-handlers"
import { ConnectionConfigStore } from "../src/config/connection-config-store"
import { CONNECTION_STATE_STORAGE_KEY, type ConnectionState } from "../src/connection/connection-state"
import { ExtensionWebSocketClient } from "../src/connection/extension-websocket-client"
import { DebugEntryStore } from "../src/debug/debug-entry-store"
import { DebugEventService } from "../src/debug/debug-event-service"
import { InteractionService } from "../src/interactions/interaction-service"
import { NetworkCaptureService } from "../src/network/network-capture-service"
import { NetworkEntryStore } from "../src/network/network-entry-store"
import { PageService } from "../src/page/page-service"
import { RefResolver } from "../src/refs/ref-resolver"
import { RefStore } from "../src/refs/ref-store"
import { PermissionService } from "../src/security/permission-service"
import { PolicyService } from "../src/security/policy-service"
import { SnapshotFormatter } from "../src/snapshots/snapshot-formatter"
import { SnapshotService } from "../src/snapshots/snapshot-service"
import { TabService } from "../src/tabs/tab-service"
import { defineBackground } from "wxt/utils/define-background"

export default defineBackground(() => {
  const cdpClient = new CdpClient()
  const sessionManager = new TabSessionManager(cdpClient)
  const refStore = new RefStore()
  const refResolver = new RefResolver(cdpClient, sessionManager, refStore)
  const interaction = new InteractionService(cdpClient, sessionManager, refResolver)
  const network = new NetworkCaptureService(cdpClient, sessionManager, new NetworkEntryStore())
  const debug = new DebugEventService(cdpClient, sessionManager, new DebugEntryStore())
  const registry = new HandlerRegistry()
  const configStore = new ConnectionConfigStore()

  registerHandlers(registry, {
    tabs: new TabService(),
    page: new PageService(cdpClient, sessionManager),
    snapshot: new SnapshotService(
      cdpClient,
      sessionManager,
      refStore,
      new SnapshotFormatter()
    ),
    interaction,
    network,
    debug,
    policy: new PolicyService(configStore)
  })

  const commandProcessor = new CommandProcessor(registry, new PermissionService(configStore))
  const websocketClient = new ExtensionWebSocketClient(
    configStore,
    commandProcessor,
    publishConnectionState
  )

  chrome.runtime.onMessage.addListener((message: unknown) => {
    if (!isRecord(message) || typeof message.type !== "string") {
      return undefined
    }
    if (message.type === "connect") {
      return websocketClient.connect()
    }
    return undefined
  })

  void websocketClient.connect().catch(() => undefined)
})

function publishConnectionState(state: ConnectionState): void {
  void chrome.storage.session.set({ [CONNECTION_STATE_STORAGE_KEY]: state })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}
