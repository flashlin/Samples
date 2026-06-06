import type { NetworkRequest } from "@page-probe/protocol"

export class NetworkEntryStore {
  readonly entries = new Map<number, Map<string, NetworkRequest>>()
  readonly captureTabs = new Set<number>()

  start(tabId: number): void {
    this.captureTabs.add(tabId)
    this.entries.set(tabId, new Map())
  }

  isCapturing(tabId: number): boolean {
    return this.captureTabs.has(tabId)
  }

  upsert(tabId: number, entry: NetworkRequest): void {
    const tabEntries = this.entries.get(tabId) ?? new Map<string, NetworkRequest>()
    tabEntries.set(entry.requestId, entry)
    while (tabEntries.size > 500) {
      const oldest = tabEntries.keys().next().value
      if (oldest) {
        tabEntries.delete(oldest)
      }
    }
    this.entries.set(tabId, tabEntries)
  }

  update(tabId: number, requestId: string, updates: Partial<NetworkRequest>): void {
    const entry = this.entries.get(tabId)?.get(requestId)
    if (entry) {
      Object.assign(entry, updates)
    }
  }

  list(tabId: number, limit = 500): NetworkRequest[] {
    return [...(this.entries.get(tabId)?.values() ?? [])].slice(-limit)
  }

  require(tabId: number, requestId: string): NetworkRequest | undefined {
    return this.entries.get(tabId)?.get(requestId)
  }
}
