import { BrowserProtocolError, type NetworkRequest } from "@page-probe/protocol"
import type { CdpClient } from "../cdp/cdp-client"
import type { TabSessionManager } from "../cdp/tab-session-manager"
import type { NetworkEntryStore } from "./network-entry-store"

const responseBodyLimit = 5 * 1024 * 1024

export class NetworkCaptureService {
  constructor(
    private readonly cdpClient: CdpClient,
    private readonly sessionManager: TabSessionManager,
    private readonly store: NetworkEntryStore
  ) {
    this.sessionManager.subscribe((session, _source, method, params) => {
      if (this.store.isCapturing(session.tabId)) {
        this.handleEvent(session.tabId, method, params)
      }
    })
  }

  async start(tabId: number): Promise<{ success: true }> {
    await this.sessionManager.require(tabId)
    this.store.start(tabId)
    return { success: true }
  }

  list(tabId: number, limit?: number): { requests: NetworkRequest[] } {
    return { requests: this.store.list(tabId, limit) }
  }

  async getResponseBody(tabId: number, requestId: string) {
    const entry = this.store.require(tabId, requestId)
    if (!entry?.finished) {
      throw new BrowserProtocolError("INVALID_PARAMS", "Response body is not available")
    }
    const response = await this.cdpClient.send<{ body: string; base64Encoded: boolean }>(
      this.sessionManager.target(tabId),
      "Network.getResponseBody",
      { requestId }
    )
    const originalSize = new TextEncoder().encode(response.body).byteLength
    return {
      body: originalSize > responseBodyLimit ? response.body.slice(0, responseBodyLimit) : response.body,
      base64Encoded: response.base64Encoded,
      truncated: originalSize > responseBodyLimit,
      originalSize
    }
  }

  private handleEvent(tabId: number, method: string, params: unknown): void {
    if (!isRecord(params) || typeof params.requestId !== "string") {
      return
    }

    if (method === "Network.requestWillBeSent" && isRecord(params.request)) {
      this.store.upsert(tabId, {
        requestId: params.requestId,
        url: readString(params.request.url),
        method: readString(params.request.method),
        ...(readOptionalString(params.type) ? { type: readOptionalString(params.type) } : {}),
        timestamp: readNumber(params.timestamp),
        finished: false
      })
    }
    if (method === "Network.responseReceived" && isRecord(params.response)) {
      this.store.update(tabId, params.requestId, {
        status: readNumber(params.response.status),
        mimeType: readString(params.response.mimeType)
      })
    }
    if (method === "Network.loadingFinished") {
      this.store.update(tabId, params.requestId, { finished: true })
    }
    if (method === "Network.loadingFailed") {
      this.store.update(tabId, params.requestId, {
        finished: true,
        failed: true,
        ...(readOptionalString(params.errorText) ? { errorText: readOptionalString(params.errorText) } : {})
      })
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : ""
}

function readOptionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined
}

function readNumber(value: unknown): number {
  return typeof value === "number" ? value : 0
}
