import { BrowserProtocolError } from "@page-probe/protocol"
import type { ExtensionConnection } from "./extension-connection"

export class ConnectionRegistry {
  readonly connections = new Map<string, ExtensionConnection>()

  register(connection: ExtensionConnection): void {
    const existing = this.connections.get(connection.clientId)
    existing?.socket.close(4001, "Replaced by a newer connection")
    this.connections.set(connection.clientId, connection)
  }

  remove(clientId: string): void {
    this.connections.delete(clientId)
  }

  require(clientId?: string): ExtensionConnection {
    if (clientId) {
      const connection = this.connections.get(clientId)
      if (connection) {
        return connection
      }
      throw new BrowserProtocolError("CLIENT_NOT_CONNECTED", `Client ${clientId} is not connected`)
    }

    if (this.connections.size === 1) {
      return this.connections.values().next().value as ExtensionConnection
    }

    throw new BrowserProtocolError(
      "CLIENT_NOT_CONNECTED",
      this.connections.size === 0 ? "No browser client is connected" : "A clientId is required"
    )
  }

  list(): Array<{ clientId: string; name: string; version: string }> {
    return [...this.connections.values()].map(({ clientId, name, version }) => ({
      clientId,
      name,
      version
    }))
  }
}
