import { afterEach, describe, expect, it } from "bun:test"
import { BrowserProtocolError } from "@page-probe/protocol"
import type { ConnectionConfigStore } from "../src/config/connection-config-store"
import { PermissionService } from "../src/security/permission-service"

const originalChrome = globalThis.chrome

afterEach(() => {
  Object.defineProperty(globalThis, "chrome", {
    configurable: true,
    value: originalChrome
  })
})

describe("PermissionService", () => {
  it("rejects debug evaluation by default", async () => {
    const service = new PermissionService(createConfigStore([], false))

    await expect(service.assertDebugEvaluateAllowed()).rejects.toMatchObject({
      code: "PERMISSION_DENIED",
      message: "Debug evaluation is disabled"
    })
  })

  it("allows debug evaluation after explicit authorization", async () => {
    const service = new PermissionService(createConfigStore([], true))

    await expect(service.assertDebugEvaluateAllowed()).resolves.toBeUndefined()
  })

  it("allows a tab when its origin is in the allowlist and Chrome permission passes", async () => {
    installChromeMock(true)
    const service = new PermissionService(createConfigStore(["https://example.com"]))

    const tab = await service.assertTabAllowed(7)

    expect(tab.id).toBe(7)
  })

  it("rejects an origin missing from the allowlist", async () => {
    installChromeMock(true)
    const service = new PermissionService(createConfigStore(["https://allowed.example"]))

    await expect(service.assertTabAllowed(7)).rejects.toMatchObject({
      code: "PERMISSION_DENIED",
      message: "Origin is not in the runtime allowlist: https://example.com"
    })
  })

  it("rejects a tab when Chrome host permission is missing", async () => {
    installChromeMock(false)
    const service = new PermissionService(createConfigStore(["https://example.com"]))

    await expect(service.assertTabAllowed(7)).rejects.toBeInstanceOf(BrowserProtocolError)
    await expect(service.assertTabAllowed(7)).rejects.toMatchObject({
      code: "PERMISSION_DENIED",
      message: "Website access is required for https://example.com/*"
    })
  })

  it("rejects Chrome internal pages", async () => {
    installChromeMock(true, "chrome://settings")
    const service = new PermissionService(createConfigStore([]))

    await expect(service.assertTabAllowed(7)).rejects.toMatchObject({
      code: "UNSUPPORTED_PAGE"
    })
  })

  it("allows a navigation target origin in the allowlist", async () => {
    installChromeMock(true)
    const service = new PermissionService(createConfigStore(["https://example.com"]))

    await expect(service.assertOriginAllowed("https://example.com")).resolves.toBeUndefined()
  })

  it("rejects a navigation target origin missing from the allowlist", async () => {
    installChromeMock(true)
    const service = new PermissionService(createConfigStore(["https://example.com"]))

    await expect(service.assertOriginAllowed("https://evil.example")).rejects.toMatchObject({
      code: "PERMISSION_DENIED",
      message: "Origin is not in the runtime allowlist: https://evil.example"
    })
  })

  it("rejects a navigation target when Chrome host permission is missing", async () => {
    installChromeMock(false)
    const service = new PermissionService(createConfigStore(["https://example.com"]))

    await expect(service.assertOriginAllowed("https://example.com")).rejects.toMatchObject({
      code: "PERMISSION_DENIED",
      message: "Website access is required for https://example.com/*"
    })
  })
})

function createConfigStore(
  allowedOrigins: string[],
  debugEvaluateEnabled = false
): ConnectionConfigStore {
  return {
    load: async () => ({
      url: "ws://127.0.0.1:17890/extension",
      token: "token",
      allowedOrigins,
      debugEvaluateEnabled
    })
  } as ConnectionConfigStore
}

function installChromeMock(permissionGranted: boolean, url = "https://example.com/page"): void {
  Object.defineProperty(globalThis, "chrome", {
    configurable: true,
    value: {
      tabs: {
        get: async (tabId: number) => ({
          id: tabId,
          active: true,
          url
        })
      },
      permissions: {
        contains: async () => permissionGranted
      }
    }
  })
}
