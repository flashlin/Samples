import { describe, expect, it } from "bun:test"
import {
  BrowserProtocolError,
  protocolVersion,
  type CommandEnvelope
} from "@page-probe/protocol"
import type { ServerWebSocket } from "bun"
import { CommandDispatcher } from "../src/command-bus/command-dispatcher"
import { PendingRequestStore } from "../src/command-bus/pending-request-store"
import { TimeoutPolicy } from "../src/command-bus/timeout-policy"
import { ConnectionRegistry } from "../src/connections/connection-registry"
import {
  ExtensionConnection,
  type ExtensionSocketData
} from "../src/connections/extension-connection"

describe("Command bus", () => {
  it("dispatches a command envelope through the only connection", async () => {
    const context = createContext()
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const result = context.dispatcher.dispatch("tabs.list", {})
    const command = connection.sentCommands[0]

    expect(command).toMatchObject({
      protocolVersion,
      type: "command",
      clientId: "client-1",
      method: "tabs.list",
      params: {}
    })
    expect(command.id).toBeString()
    expect(new Date(command.deadline).getTime()).toBeGreaterThan(Date.now())

    context.pendingRequestStore.settleUnknown(createSuccessResult(command.id, {
      tabs: []
    }))

    expect(await result).toEqual({ tabs: [] })
  })

  it("correlates results by command id", async () => {
    const context = createContext()
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const firstResult = context.dispatcher.dispatch("tabs.list", {})
    const secondResult = context.dispatcher.dispatch("tabs.getActive", {})
    const [firstCommand, secondCommand] = connection.sentCommands
    const activeTab = {
      id: 7,
      active: true,
      title: "Active tab",
      url: "https://example.com"
    }

    context.pendingRequestStore.settleUnknown(createSuccessResult(secondCommand.id, activeTab))

    expect(await secondResult).toEqual(activeTab)
    expect(context.pendingRequestStore.requests.has(firstCommand.id)).toBe(true)

    context.pendingRequestStore.settleUnknown(createSuccessResult(firstCommand.id, {
      tabs: []
    }))

    expect(await firstResult).toEqual({ tabs: [] })
  })

  it("rejects an error result", async () => {
    const context = createContext()
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const result = context.dispatcher.dispatch("tabs.list", {})
    const command = connection.sentCommands[0]

    context.pendingRequestStore.settleUnknown({
      protocolVersion,
      type: "result",
      id: command.id,
      ok: false,
      error: {
        code: "INTERNAL_ERROR",
        message: "Extension command failed",
        details: { reason: "test" }
      }
    })

    await expect(result).rejects.toMatchObject({
      name: "BrowserProtocolError",
      code: "INTERNAL_ERROR",
      message: "Extension command failed",
      details: { reason: "test" }
    })
  })

  it("rejects a command after its timeout", async () => {
    const context = createContext(10)
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const result = context.dispatcher.dispatch("tabs.list", {})

    await expect(result).rejects.toMatchObject({
      name: "BrowserProtocolError",
      code: "ACTION_TIMEOUT",
      message: "Command tabs.list timed out"
    })
    expect(context.pendingRequestStore.requests.size).toBe(0)
  })

  it("rejects pending commands when their client disconnects", async () => {
    const context = createContext()
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const result = context.dispatcher.dispatch("tabs.list", {})

    context.pendingRequestStore.rejectClient("client-1")

    await expect(result).rejects.toMatchObject({
      name: "BrowserProtocolError",
      code: "CLIENT_NOT_CONNECTED",
      message: "Browser client disconnected"
    })
    expect(context.pendingRequestStore.requests.size).toBe(0)
  })

  it("ignores a result received after timeout", async () => {
    const context = createContext(10)
    const connection = createConnection("client-1")
    context.connectionRegistry.register(connection.value)

    const result = context.dispatcher.dispatch("tabs.list", {})
    const command = connection.sentCommands[0]

    await expect(result).rejects.toBeInstanceOf(BrowserProtocolError)

    expect(() => {
      context.pendingRequestStore.settleUnknown(createSuccessResult(command.id, {
        tabs: []
      }))
    }).not.toThrow()
    expect(context.pendingRequestStore.requests.size).toBe(0)
  })

  it("requires clientId when multiple clients are connected", () => {
    const context = createContext()
    context.connectionRegistry.register(createConnection("client-1").value)
    context.connectionRegistry.register(createConnection("client-2").value)

    expect(() => context.dispatcher.dispatch("tabs.list", {})).toThrow(
      expect.objectContaining({
        name: "BrowserProtocolError",
        code: "CLIENT_NOT_CONNECTED",
        message: "A clientId is required"
      })
    )
  })
})

function createContext(timeoutMs = 1_000): {
  connectionRegistry: ConnectionRegistry
  pendingRequestStore: PendingRequestStore
  dispatcher: CommandDispatcher
} {
  const connectionRegistry = new ConnectionRegistry()
  const pendingRequestStore = new PendingRequestStore()
  const timeoutPolicy = new TimeoutPolicy(timeoutMs)

  return {
    connectionRegistry,
    pendingRequestStore,
    dispatcher: new CommandDispatcher(connectionRegistry, pendingRequestStore, timeoutPolicy)
  }
}

function createConnection(clientId: string): {
  value: ExtensionConnection
  sentCommands: CommandEnvelope[]
} {
  const sentCommands: CommandEnvelope[] = []
  const socket = {
    send(message: string) {
      sentCommands.push(JSON.parse(message) as CommandEnvelope)
      return message.length
    },
    close() {}
  } as unknown as ServerWebSocket<ExtensionSocketData>

  return {
    value: new ExtensionConnection(clientId, "Chrome", "1.0.0", socket),
    sentCommands
  }
}

function createSuccessResult(id: string, result: unknown): unknown {
  return {
    protocolVersion,
    type: "result",
    id,
    ok: true,
    result
  }
}
