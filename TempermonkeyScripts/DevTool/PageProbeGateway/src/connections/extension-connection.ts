import type { ServerWebSocket } from "bun"
import type { CommandEnvelope } from "@page-probe/protocol"

export type ExtensionSocketData = {
  authenticated: boolean
  clientId?: string
  connectedAt: number
  lastPongAt: number
  keepAliveTimer?: ReturnType<typeof setInterval> | undefined
}

export class ExtensionConnection {
  readonly clientId: string
  readonly name: string
  readonly version: string
  readonly socket: ServerWebSocket<ExtensionSocketData>

  constructor(
    clientId: string,
    name: string,
    version: string,
    socket: ServerWebSocket<ExtensionSocketData>
  ) {
    this.clientId = clientId
    this.name = name
    this.version = version
    this.socket = socket
  }

  send(command: CommandEnvelope): void {
    this.socket.send(JSON.stringify(command))
  }
}
