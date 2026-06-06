import { BrowserProtocolError } from "@page-probe/protocol"
import type { ConnectionConfigStore } from "../config/connection-config-store"

export class PermissionService {
  constructor(private readonly configStore: ConnectionConfigStore) {}

  async assertDebugEvaluateAllowed(): Promise<void> {
    const config = await this.configStore.load()
    if (!config.debugEvaluateEnabled) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Debug evaluation is disabled")
    }
  }

  async assertTabAllowed(tabId: number): Promise<chrome.tabs.Tab> {
    let tab: chrome.tabs.Tab
    try {
      tab = await chrome.tabs.get(tabId)
    } catch {
      throw new BrowserProtocolError("TAB_NOT_FOUND", `Tab ${tabId} was not found`)
    }

    if (!tab.url || !isSupportedUrl(tab.url)) {
      throw new BrowserProtocolError("UNSUPPORTED_PAGE", "This page cannot be controlled")
    }

    const origin = new URL(tab.url).origin
    const config = await this.configStore.load()
    if (!config.allowedOrigins.includes(origin)) {
      throw new BrowserProtocolError("PERMISSION_DENIED", `Origin is not in the runtime allowlist: ${origin}`)
    }

    const originPattern = `${origin}/*`
    const allowed = await chrome.permissions.contains({ origins: [originPattern] })
    if (!allowed) {
      throw new BrowserProtocolError("PERMISSION_DENIED", `Website access is required for ${originPattern}`)
    }
    return tab
  }
}

function isSupportedUrl(url: string): boolean {
  try {
    const protocol = new URL(url).protocol
    return protocol === "http:" || protocol === "https:"
  } catch {
    return false
  }
}
