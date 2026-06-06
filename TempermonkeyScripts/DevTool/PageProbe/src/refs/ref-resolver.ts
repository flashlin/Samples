import { BrowserProtocolError, type RefTarget } from "@page-probe/protocol"
import type { CdpClient } from "../cdp/cdp-client"
import type { TabSessionManager } from "../cdp/tab-session-manager"
import type { InternalRefEntry, RefStore } from "./ref-store"
import type { AxNode } from "../snapshots/ax-tree-normalizer"

type ResolveNodeResponse = { object: { objectId?: string } }
type AxTreeResponse = { nodes: AxNode[] }

export type ResolvedElement = {
  entry: InternalRefEntry
  backendNodeId: number
  objectId: string
}

export class RefResolver {
  constructor(
    private readonly cdpClient: CdpClient,
    private readonly sessionManager: TabSessionManager,
    private readonly refStore: RefStore
  ) {}

  async resolve(target: RefTarget): Promise<ResolvedElement> {
    const session = await this.sessionManager.require(target.tabId)
    const snapshot = this.refStore.get(target.tabId, target.snapshotId)
    if (!snapshot) {
      throw new BrowserProtocolError("SNAPSHOT_NOT_FOUND", "Snapshot was not found")
    }
    if (snapshot.documentEpoch !== session.documentEpoch) {
      throw new BrowserProtocolError("STALE_REF", "Reference belongs to an older document")
    }

    const refId = target.ref.startsWith("@") ? target.ref.slice(1) : target.ref
    const entry = snapshot.refs.get(refId)
    if (!entry) {
      throw new BrowserProtocolError("UNKNOWN_REF", `Reference ${target.ref} was not found`)
    }

    if (entry.backendNodeId !== undefined) {
      const fastPath = await this.tryResolve(entry)
      if (fastPath) {
        return fastPath
      }
    }

    return this.recover(entry)
  }

  private async tryResolve(entry: InternalRefEntry): Promise<ResolvedElement | undefined> {
    try {
      const backendNodeId = entry.backendNodeId
      if (backendNodeId === undefined) {
        return undefined
      }
      const response = await this.cdpClient.send<ResolveNodeResponse>(
        this.sessionManager.target(entry.tabId, entry.sessionId),
        "DOM.resolveNode",
        { backendNodeId }
      )
      return response.object.objectId
        ? { entry, backendNodeId, objectId: response.object.objectId }
        : undefined
    } catch {
      return undefined
    }
  }

  private async recover(entry: InternalRefEntry): Promise<ResolvedElement> {
    const target = this.sessionManager.target(entry.tabId, entry.sessionId)
    const tree = await this.cdpClient.send<AxTreeResponse>(target, "Accessibility.getFullAXTree")
    const candidates = tree.nodes.filter((node) => (
      readValue(node.role) === entry.role
      && readValue(node.name) === entry.name
      && node.backendDOMNodeId !== undefined
    ))
    const index = entry.nth ?? 0
    if (entry.nth === undefined && candidates.length > 1) {
      throw new BrowserProtocolError("AMBIGUOUS_REF", "Multiple elements match this reference")
    }
    const candidate = candidates[index]
    if (!candidate?.backendDOMNodeId) {
      throw new BrowserProtocolError("STALE_REF", "Reference could not be recovered")
    }
    entry.backendNodeId = candidate.backendDOMNodeId
    const resolved = await this.tryResolve(entry)
    if (!resolved) {
      throw new BrowserProtocolError("STALE_REF", "Reference could not be resolved")
    }
    return resolved
  }
}

function readValue(value: { value?: unknown } | undefined): string {
  return typeof value?.value === "string" ? value.value : ""
}
