import { BrowserProtocolError, type BrowserMethod } from "@page-probe/protocol"

export type CommandHandler = (params: never, clientId: string) => Promise<unknown> | unknown

export class HandlerRegistry {
  readonly handlers = new Map<BrowserMethod, CommandHandler>()

  register(method: BrowserMethod, handler: CommandHandler): void {
    this.handlers.set(method, handler)
  }

  require(method: BrowserMethod): CommandHandler {
    const handler = this.handlers.get(method)
    if (!handler) {
      throw new BrowserProtocolError("UNKNOWN_COMMAND", `No handler is registered for ${method}`)
    }
    return handler
  }
}
