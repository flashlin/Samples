import { z } from "zod"

export const browserErrorCodes = [
  "UNKNOWN_COMMAND",
  "INVALID_PARAMS",
  "CLIENT_NOT_CONNECTED",
  "TAB_NOT_FOUND",
  "UNSUPPORTED_PAGE",
  "DEBUGGER_ATTACH_FAILED",
  "SNAPSHOT_NOT_FOUND",
  "UNKNOWN_REF",
  "STALE_REF",
  "AMBIGUOUS_REF",
  "ELEMENT_NOT_VISIBLE",
  "ELEMENT_NOT_ENABLED",
  "ACTION_TIMEOUT",
  "PERMISSION_DENIED",
  "OUTPUT_TRUNCATED",
  "INTERNAL_ERROR"
] as const

export const browserErrorCodeSchema = z.enum(browserErrorCodes)

export const browserErrorSchema = z.object({
  code: browserErrorCodeSchema,
  message: z.string(),
  details: z.unknown().optional()
})

export type BrowserErrorCode = z.infer<typeof browserErrorCodeSchema>
export type BrowserError = z.infer<typeof browserErrorSchema>

export class BrowserProtocolError extends Error {
  readonly code: BrowserErrorCode
  readonly details?: unknown

  constructor(code: BrowserErrorCode, message: string, details?: unknown) {
    super(message)
    this.name = "BrowserProtocolError"
    this.code = code
    this.details = details
  }
}
