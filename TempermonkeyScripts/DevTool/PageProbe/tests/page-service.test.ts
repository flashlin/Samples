import { describe, expect, it } from "bun:test"
import type { CdpClient } from "../src/cdp/cdp-client"
import type { TabSessionManager } from "../src/cdp/tab-session-manager"
import { PageService } from "../src/page/page-service"

type SendCall = { target: unknown; method: string; params: unknown }

function createService() {
  const calls: SendCall[] = []
  const required: number[] = []
  const cdpClient = {
    send: async (target: unknown, method: string, params: unknown) => {
      calls.push({ target, method, params })
      return {}
    }
  } as unknown as CdpClient
  const sessionManager = {
    require: async (tabId: number) => {
      required.push(tabId)
    },
    target: (tabId: number) => ({ tabId })
  } as unknown as TabSessionManager
  return { service: new PageService(cdpClient, sessionManager), calls, required }
}

describe("PageService navigation", () => {
  it("navigates a tab via Page.navigate", async () => {
    const { service, calls, required } = createService()

    const result = await service.navigate(7, "https://example.com/path")

    expect(required).toEqual([7])
    expect(calls).toEqual([
      { target: { tabId: 7 }, method: "Page.navigate", params: { url: "https://example.com/path" } }
    ])
    expect(result).toEqual({ success: true })
  })

  it("reloads a tab via Page.reload", async () => {
    const { service, calls, required } = createService()

    const result = await service.reload(7)

    expect(required).toEqual([7])
    expect(calls).toEqual([
      { target: { tabId: 7 }, method: "Page.reload", params: {} }
    ])
    expect(result).toEqual({ success: true })
  })

  it("passes ignoreCache through to Page.reload", async () => {
    const { service, calls } = createService()

    await service.reload(7, true)

    expect(calls).toEqual([
      { target: { tabId: 7 }, method: "Page.reload", params: { ignoreCache: true } }
    ])
  })

  it("goes back through session history", async () => {
    const { service, calls, required } = createService()

    const result = await service.goBack(7)

    expect(required).toEqual([7])
    expect(calls).toEqual([
      { target: { tabId: 7 }, method: "Runtime.evaluate", params: { expression: "history.back()" } }
    ])
    expect(result).toEqual({ success: true })
  })

  it("goes forward through session history", async () => {
    const { service, calls, required } = createService()

    const result = await service.goForward(7)

    expect(required).toEqual([7])
    expect(calls).toEqual([
      { target: { tabId: 7 }, method: "Runtime.evaluate", params: { expression: "history.forward()" } }
    ])
    expect(result).toEqual({ success: true })
  })
})
