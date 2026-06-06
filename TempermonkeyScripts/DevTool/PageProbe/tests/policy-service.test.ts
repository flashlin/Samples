import { describe, expect, it } from "bun:test"
import type { ConnectionConfigStore } from "../src/config/connection-config-store"
import { PolicyService } from "../src/security/policy-service"

describe("PolicyService", () => {
  it("returns the configured allowed origins", async () => {
    const service = new PolicyService({
      load: async () => ({
        url: "ws://127.0.0.1:17890/extension",
        token: "token",
        allowedOrigins: ["https://example.com", "https://app.example.com"],
        debugEvaluateEnabled: false
      })
    } as unknown as ConnectionConfigStore)

    expect(await service.listAllowedOrigins()).toEqual({
      origins: ["https://example.com", "https://app.example.com"]
    })
  })

  it("returns an empty list when no origins are allowed", async () => {
    const service = new PolicyService({
      load: async () => ({
        url: "ws://127.0.0.1:17890/extension",
        token: "token",
        allowedOrigins: [],
        debugEvaluateEnabled: false
      })
    } as unknown as ConnectionConfigStore)

    expect(await service.listAllowedOrigins()).toEqual({ origins: [] })
  })
})
