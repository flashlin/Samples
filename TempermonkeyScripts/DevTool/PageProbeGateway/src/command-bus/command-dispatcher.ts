import {
  browserCommandParamSchemas,
  protocolVersion,
  type BrowserCommandParams,
  type BrowserCommandResults,
  type BrowserMethod,
  type CommandEnvelope
} from "@page-probe/protocol"
import type { ConnectionRegistry } from "../connections/connection-registry"
import type { PendingRequestStore } from "./pending-request-store"
import type { TimeoutPolicy } from "./timeout-policy"

export class CommandDispatcher {
  constructor(
    private readonly connectionRegistry: ConnectionRegistry,
    private readonly pendingRequestStore: PendingRequestStore,
    private readonly timeoutPolicy: TimeoutPolicy
  ) {}

  dispatch<TMethod extends BrowserMethod>(
    method: TMethod,
    params: BrowserCommandParams[TMethod],
    clientId?: string
  ): Promise<BrowserCommandResults[TMethod]> {
    const validatedParams = browserCommandParamSchemas[method].parse(params) as BrowserCommandParams[TMethod]
    const connection = this.connectionRegistry.require(clientId)
    const id = crypto.randomUUID()
    const timeoutMs = this.timeoutPolicy.for(method)
    const command: CommandEnvelope<TMethod> = {
      protocolVersion,
      type: "command",
      id,
      clientId: connection.clientId,
      method,
      params: validatedParams,
      deadline: new Date(Date.now() + timeoutMs).toISOString()
    }

    const result = this.pendingRequestStore.create(id, connection.clientId, method, timeoutMs)
    connection.send(command)
    return result
  }
}
