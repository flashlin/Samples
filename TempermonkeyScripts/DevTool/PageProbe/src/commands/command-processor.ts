import {
  BrowserProtocolError,
  protocolVersion,
  type CommandEnvelope,
  type ErrorEnvelope,
  type ResultEnvelope
} from "@page-probe/protocol"
import type { PermissionService } from "../security/permission-service"
import type { HandlerRegistry } from "./handler-registry"

export class CommandProcessor {
  constructor(
    private readonly handlerRegistry: HandlerRegistry,
    private readonly permissionService: PermissionService
  ) {}

  async execute(command: CommandEnvelope): Promise<ResultEnvelope | ErrorEnvelope> {
    try {
      if (Date.parse(command.deadline) <= Date.now()) {
        throw new BrowserProtocolError("ACTION_TIMEOUT", "Command deadline has expired")
      }
      if (command.method === "debug.evaluate") {
        await this.permissionService.assertDebugEvaluateAllowed()
      }
      const tabId = readTabId(command.params)
      if (tabId !== undefined) {
        await this.permissionService.assertTabAllowed(tabId)
      }
      const handler = this.handlerRegistry.require(command.method)
      const result = await handler(command.params as never, command.clientId)
      return {
        protocolVersion,
        type: "result",
        id: command.id,
        ok: true,
        result
      } as ResultEnvelope
    } catch (error) {
      const protocolError = toProtocolError(error)
      return {
        protocolVersion,
        type: "result",
        id: command.id,
        ok: false,
        error: {
          code: protocolError.code,
          message: protocolError.message,
          ...(protocolError.details === undefined ? {} : { details: protocolError.details })
        }
      }
    }
  }
}

function readTabId(params: unknown): number | undefined {
  return typeof params === "object"
    && params !== null
    && "tabId" in params
    && typeof params.tabId === "number"
    ? params.tabId
    : undefined
}

function toProtocolError(error: unknown): BrowserProtocolError {
  return error instanceof BrowserProtocolError
    ? error
    : new BrowserProtocolError(
      "INTERNAL_ERROR",
      error instanceof Error ? error.message : "Unknown extension error"
    )
}
