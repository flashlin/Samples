import { z } from "zod"
import { parseCommandEnvelope, parseResultEnvelope, type CommandEnvelope, type ErrorEnvelope, type ResultEnvelope } from "./envelopes"
import { BrowserProtocolError } from "./errors"
import type { BrowserMethod } from "./methods"

export function parseJsonPayload(payload: string): unknown {
  try {
    return JSON.parse(payload) as unknown
  } catch {
    throw new BrowserProtocolError("INVALID_PARAMS", "Invalid JSON payload")
  }
}

export function serializeEnvelope(envelope: CommandEnvelope | ResultEnvelope | ErrorEnvelope): string {
  return JSON.stringify(envelope)
}

export function parseCommandPayload(payload: string): CommandEnvelope {
  try {
    return parseCommandEnvelope(parseJsonPayload(payload))
  } catch (error) {
    if (error instanceof BrowserProtocolError) {
      throw error
    }
    if (error instanceof z.ZodError) {
      throw new BrowserProtocolError("INVALID_PARAMS", "Invalid command payload", z.flattenError(error))
    }
    throw error
  }
}

export function parseResultPayload<TMethod extends BrowserMethod>(
  method: TMethod,
  payload: string
): ResultEnvelope<TMethod> | ErrorEnvelope {
  try {
    return parseResultEnvelope(method, parseJsonPayload(payload))
  } catch (error) {
    if (error instanceof BrowserProtocolError) {
      throw error
    }
    if (error instanceof z.ZodError) {
      throw new BrowserProtocolError("INVALID_PARAMS", "Invalid result payload", z.flattenError(error))
    }
    throw error
  }
}
