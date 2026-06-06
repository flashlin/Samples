import { BrowserProtocolError } from "@page-probe/protocol"
import { ZodError } from "zod"

export function mapErrorToResponse(error: unknown): Response {
  if (error instanceof BrowserProtocolError) {
    const status = error.code === "PERMISSION_DENIED"
      ? 403
      : error.code === "TAB_NOT_FOUND" || error.code === "SNAPSHOT_NOT_FOUND"
        ? 404
        : error.code === "CLIENT_NOT_CONNECTED"
          ? 503
          : 400

    return Response.json({
      error: {
        code: error.code,
        message: error.message,
        details: error.details
      }
    }, { status })
  }

  if (error instanceof ZodError) {
    return Response.json({
      error: {
        code: "INVALID_PARAMS",
        message: "Request validation failed",
        details: error.flatten()
      }
    }, { status: 400 })
  }

  return Response.json({
    error: {
      code: "INTERNAL_ERROR",
      message: "Internal server error"
    }
  }, { status: 500 })
}
