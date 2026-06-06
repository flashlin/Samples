import { describe, expect, it } from "bun:test"
import {
  BrowserProtocolError,
  parseCommandEnvelope,
  parseCommandPayload,
  parseResultEnvelope,
  parseResultPayload,
  protocolVersion,
  type BrowserMethod
} from "../src"

const deadline = "2026-06-06T12:00:00.000Z"

const validCommands = [
  {
    name: "tabs list",
    method: "tabs.list",
    params: {}
  },
  {
    name: "page snapshot",
    method: "page.snapshot",
    params: { tabId: 7, compact: true }
  },
  {
    name: "element fill",
    method: "element.fill",
    params: { tabId: 7, snapshotId: "snapshot-1", ref: "@e2", value: "hello" }
  },
  {
    name: "debug evaluate",
    method: "debug.evaluate",
    params: { tabId: 7, expression: "document.title", returnByValue: true }
  }
] as const

const validResults: ReadonlyArray<{
  name: string
  method: BrowserMethod
  result: unknown
}> = [
  {
    name: "tabs list",
    method: "tabs.list",
    result: { tabs: [] }
  },
  {
    name: "page text",
    method: "page.getText",
    result: { text: "Example", truncated: false }
  },
  {
    name: "element click",
    method: "element.click",
    result: { success: true }
  },
  {
    name: "network response body",
    method: "network.getResponseBody",
    result: {
      body: "response",
      base64Encoded: false,
      truncated: false,
      originalSize: 8
    }
  }
]

const invalidCommands = [
  {
    name: "unknown method",
    input: {
      protocolVersion,
      type: "command",
      id: "request-1",
      clientId: "chrome-default",
      method: "unknown.method",
      params: {},
      deadline
    }
  },
  {
    name: "missing required params",
    input: {
      protocolVersion,
      type: "command",
      id: "request-1",
      clientId: "chrome-default",
      method: "element.click",
      params: { tabId: 7 },
      deadline
    }
  },
  {
    name: "protocol version mismatch",
    input: {
      protocolVersion: protocolVersion + 1,
      type: "command",
      id: "request-1",
      clientId: "chrome-default",
      method: "tabs.list",
      params: {},
      deadline
    }
  }
] as const

const mismatchedResults: ReadonlyArray<{
  name: string
  method: BrowserMethod
  result: unknown
}> = [
  {
    name: "tabs list without tabs",
    method: "tabs.list",
    result: { entries: [] }
  },
  {
    name: "action with unsuccessful result",
    method: "element.click",
    result: { success: false }
  },
  {
    name: "text result without truncation metadata",
    method: "page.getText",
    result: { text: "Example" }
  }
]

describe("command envelopes", () => {
  for (const testCase of validCommands) {
    it(`parses valid ${testCase.name} command`, () => {
      const command = parseCommandEnvelope({
        protocolVersion,
        type: "command",
        id: "request-1",
        clientId: "chrome-default",
        method: testCase.method,
        params: testCase.params,
        deadline
      })

      expect(command.method).toBe(testCase.method)
      expect(command.params).toEqual(testCase.params)
    })
  }

  for (const testCase of invalidCommands) {
    it(`rejects ${testCase.name}`, () => {
      expect(() => parseCommandEnvelope(testCase.input)).toThrow()
    })
  }
})

describe("result envelopes", () => {
  for (const testCase of validResults) {
    it(`parses valid ${testCase.name} result`, () => {
      const result = parseResultEnvelope(testCase.method, {
        protocolVersion,
        type: "result",
        id: "request-1",
        ok: true,
        result: testCase.result
      })

      expect(result).toEqual({
        protocolVersion,
        type: "result",
        id: "request-1",
        ok: true,
        result: testCase.result
      })
    })
  }

  it("parses a valid error result", () => {
    const result = parseResultEnvelope("tabs.list", {
      protocolVersion,
      type: "result",
      id: "request-1",
      ok: false,
      error: {
        code: "TAB_NOT_FOUND",
        message: "The tab was not found",
        details: { tabId: 7 }
      }
    })

    expect(result).toEqual({
      protocolVersion,
      type: "result",
      id: "request-1",
      ok: false,
      error: {
        code: "TAB_NOT_FOUND",
        message: "The tab was not found",
        details: { tabId: 7 }
      }
    })
  })

  for (const testCase of mismatchedResults) {
    it(`rejects ${testCase.name}`, () => {
      expect(() => parseResultEnvelope(testCase.method, {
        protocolVersion,
        type: "result",
        id: "request-1",
        ok: true,
        result: testCase.result
      })).toThrow()
    })
  }
})

describe("JSON payloads", () => {
  const invalidJsonPayloads = [
    "",
    "{",
    "{\"type\":\"command\",}"
  ]

  for (const payload of invalidJsonPayloads) {
    it(`rejects invalid JSON ${JSON.stringify(payload)}`, () => {
      expect(() => parseCommandPayload(payload)).toThrow("Invalid JSON payload")
    })
  }

  it("converts invalid command schema errors to protocol errors", () => {
    const payload = JSON.stringify({
      protocolVersion,
      type: "command",
      id: "request-1",
      clientId: "chrome-default",
      method: "element.click",
      params: { tabId: 7 },
      deadline
    })

    try {
      parseCommandPayload(payload)
      throw new Error("Expected command parsing to fail")
    } catch (error) {
      expect(error).toBeInstanceOf(BrowserProtocolError)
      expect((error as BrowserProtocolError).code).toBe("INVALID_PARAMS")
      expect((error as BrowserProtocolError).message).toBe("Invalid command payload")
    }
  })

  it("converts result schema mismatches to protocol errors", () => {
    const payload = JSON.stringify({
      protocolVersion,
      type: "result",
      id: "request-1",
      ok: true,
      result: { success: false }
    })

    try {
      parseResultPayload("element.click", payload)
      throw new Error("Expected result parsing to fail")
    } catch (error) {
      expect(error).toBeInstanceOf(BrowserProtocolError)
      expect((error as BrowserProtocolError).code).toBe("INVALID_PARAMS")
      expect((error as BrowserProtocolError).message).toBe("Invalid result payload")
    }
  })
})
