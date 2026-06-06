import {
  authResultSchema,
  heartbeatSchema,
  parseCommandEnvelope,
  protocolVersion
} from "@page-probe/protocol"
import type { CommandProcessor } from "../commands/command-processor"
import type { ConnectionConfigStore } from "../config/connection-config-store"
import type { ConnectionState } from "./connection-state"

const RECONNECT_DELAY_MS = 10_000

export class ExtensionWebSocketClient {
  private socket: WebSocket | undefined
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined
  private stopped = true

  constructor(
    private readonly configStore: ConnectionConfigStore,
    private readonly commandProcessor: CommandProcessor,
    private readonly onStateChange: (state: ConnectionState) => void
  ) {}

  async connect(): Promise<void> {
    this.stopped = false
    clearTimeout(this.reconnectTimer)
    this.socket?.close()
    this.onStateChange("connecting")
    const config = await this.configStore.load()
    if (!config.token) {
      this.onStateChange("disconnected")
      this.scheduleReconnect()
      return
    }
    const socket = new WebSocket(config.url)
    this.socket = socket
    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        protocolVersion,
        type: "auth",
        token: config.token,
        client: {
          clientId: `chrome-${chrome.runtime.id}`,
          name: "PageProbe",
          version: chrome.runtime.getManifest().version
        }
      }))
    })
    socket.addEventListener("message", (event) => {
      void this.handleMessage(event.data)
    })
    socket.addEventListener("close", () => {
      this.socket = undefined
      this.onStateChange("disconnected")
      if (!this.stopped) {
        this.scheduleReconnect()
      }
    })
  }

  private async handleMessage(data: unknown): Promise<void> {
    if (typeof data !== "string" || !this.socket) {
      return
    }
    const input = JSON.parse(data) as unknown
    const authResult = authResultSchema.safeParse(input)
    if (authResult.success) {
      if (authResult.data.ok) {
        this.onStateChange("connected")
      } else {
        this.socket.close()
      }
      return
    }
    const heartbeat = heartbeatSchema.safeParse(input)
    if (heartbeat.success && heartbeat.data.type === "ping") {
      this.socket.send(JSON.stringify({ ...heartbeat.data, type: "pong" }))
      return
    }
    const command = parseCommandEnvelope(input)
    const result = await this.commandProcessor.execute(command)
    this.socket.send(JSON.stringify(result))
  }

  private scheduleReconnect(): void {
    clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      void this.connect()
    }, RECONNECT_DELAY_MS)
  }
}
