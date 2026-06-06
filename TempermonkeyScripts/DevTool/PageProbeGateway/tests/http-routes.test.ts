import { describe, expect, it } from "bun:test"
import { BearerAuth } from "../src/auth/bearer-auth"
import { RequestOriginGuard } from "../src/auth/request-origin-guard"
import { createClientsRoute } from "../src/http/routes/clients-route"
import { createHealthRoute } from "../src/http/routes/health-route"

const token = "test-token"
const host = "127.0.0.1:17890"
const origin = "http://127.0.0.1:17890"

describe("HTTP routes", () => {
  const security = {
    bearerAuth: new BearerAuth(token),
    requestOriginGuard: new RequestOriginGuard([origin], [host])
  }

  it("returns health for an authenticated request", async () => {
    const response = createHealthRoute(security)(createRequest())

    expect(response.status).toBe(200)
    expect(await response.json()).toEqual({ status: "ok" })
  })

  it("returns connected clients", async () => {
    const clients = [
      {
        clientId: "chrome-default",
        name: "Chrome",
        version: "1.0.0"
      }
    ]
    const route = createClientsRoute({
      ...security,
      connectionRegistry: {
        list: () => clients
      }
    })

    const response = route(createRequest())

    expect(response.status).toBe(200)
    expect(await response.json()).toEqual({ clients })
  })

  it("denies a missing bearer token", async () => {
    const response = createHealthRoute(security)(createRequest({
      authorization: undefined
    }))

    expect(response.status).toBe(401)
    expect(response.headers.get("www-authenticate")).toBe("Bearer")
    expect(await response.json()).toEqual({
      error: {
        code: "PERMISSION_DENIED",
        message: "Bearer token is required"
      }
    })
  })

  it("denies an invalid bearer token", async () => {
    const response = createHealthRoute(security)(createRequest({
      authorization: "Bearer invalid-token"
    }))

    expect(response.status).toBe(401)
    expect(await response.json()).toEqual({
      error: {
        code: "PERMISSION_DENIED",
        message: "Bearer token is invalid"
      }
    })
  })

  it("denies an invalid host", async () => {
    const response = createHealthRoute(security)(createRequest({
      host: "attacker.example"
    }))

    expect(response.status).toBe(403)
    expect(await response.json()).toEqual({
      error: {
        code: "PERMISSION_DENIED",
        message: "Request host is not allowed"
      }
    })
  })

  it("denies an invalid origin", async () => {
    const response = createHealthRoute(security)(createRequest({
      origin: "https://attacker.example"
    }))

    expect(response.status).toBe(403)
    expect(await response.json()).toEqual({
      error: {
        code: "PERMISSION_DENIED",
        message: "Request origin is not allowed"
      }
    })
  })
})

function createRequest(overrides: {
  authorization?: string
  host?: string
  origin?: string
} = {}): Request {
  const headers = new Headers()
  const authorization = "authorization" in overrides
    ? overrides.authorization
    : `Bearer ${token}`

  if (authorization) {
    headers.set("authorization", authorization)
  }
  headers.set("host", overrides.host ?? host)
  headers.set("origin", overrides.origin ?? origin)

  return new Request(`http://${host}/health`, {
    headers
  })
}
