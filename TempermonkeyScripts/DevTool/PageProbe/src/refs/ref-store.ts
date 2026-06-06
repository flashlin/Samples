import type { PublicRef } from "@page-probe/protocol"

export type InternalRefEntry = PublicRef & {
  refId: string
  snapshotId: string
  clientId: string
  tabId: number
  documentEpoch: number
  sessionId?: string
  backendNodeId?: number
  selectorFallback?: string
  createdAt: number
}

export type StoredSnapshot = {
  snapshotId: string
  tabId: number
  documentEpoch: number
  refs: Map<string, InternalRefEntry>
}

export class RefStore {
  readonly snapshots = new Map<number, StoredSnapshot>()

  replace(snapshot: StoredSnapshot): void {
    this.snapshots.set(snapshot.tabId, snapshot)
  }

  get(tabId: number, snapshotId: string): StoredSnapshot | undefined {
    const snapshot = this.snapshots.get(tabId)
    return snapshot?.snapshotId === snapshotId ? snapshot : undefined
  }

  clear(tabId: number): void {
    this.snapshots.delete(tabId)
  }
}
