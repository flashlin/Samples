import { BrowserProtocolError } from "@page-probe/protocol"
import type { BearerAuth } from "../auth/bearer-auth"
import type { RequestOriginGuard } from "../auth/request-origin-guard"

export type HttpRouteSecurity = {
  bearerAuth: Pick<BearerAuth, "assertRequest">
  requestOriginGuard: Pick<RequestOriginGuard, "assertRequest">
}

export function handleAuthenticatedJsonRequest(
  request: Request,
  security: HttpRouteSecurity,
  createBody: () => unknown
): Response {
  const originError = assertRequest(() => security.requestOriginGuard.assertRequest(request))
  if (originError) {
    return errorResponse(originError, 403)
  }

  const authenticationError = assertRequest(() => security.bearerAuth.assertRequest(request))
  if (authenticationError) {
    return errorResponse(authenticationError, 401, {
      "www-authenticate": "Bearer"
    })
  }

  return jsonResponse(createBody())
}

export function jsonResponse(body: unknown, status = 200, headers?: HeadersInit): Response {
  return Response.json(body, headers ? { status, headers } : { status })
}

function assertRequest(assertion: () => void): BrowserProtocolError | undefined {
  try {
    assertion()
    return undefined
  } catch (error) {
    if (error instanceof BrowserProtocolError) {
      return error
    }
    throw error
  }
}

function errorResponse(error: BrowserProtocolError, status: number, headers?: HeadersInit): Response {
  return jsonResponse({
    error: {
      code: error.code,
      message: error.message,
      ...(error.details === undefined ? {} : { details: error.details })
    }
  }, status, headers)
}
