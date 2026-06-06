import {
  BrowserProtocolError,
  type PublicRef,
  type SnapshotRequest,
  type SnapshotResult
} from "@page-probe/protocol"
import type { CdpClient } from "../cdp/cdp-client"
import type { TabSessionManager } from "../cdp/tab-session-manager"
import { RefAllocator } from "../refs/ref-allocator"
import type { InternalRefEntry, RefStore } from "../refs/ref-store"
import { normalizeAxTree, type AxNode, type NormalizedAxNode } from "./ax-tree-normalizer"
import type { SnapshotFormatter } from "./snapshot-formatter"

type AxTreeResponse = { nodes: AxNode[] }
type RuntimeEvaluateResponse = { result: { objectId?: string; subtype?: string } }
type DomDescribeResponse = { node: { backendNodeId?: number } }
type DecoratedNode = NormalizedAxNode & { refId?: string }
type SnapshotRefs = {
  decorated: DecoratedNode[]
  internal: Map<string, InternalRefEntry>
  public: Record<string, PublicRef>
}

export class SnapshotService {
  constructor(
    private readonly cdpClient: CdpClient,
    private readonly sessionManager: TabSessionManager,
    private readonly refStore: RefStore,
    private readonly formatter: SnapshotFormatter
  ) {
    this.sessionManager.subscribe((session, _source, method, params) => {
      if (method === "Page.frameNavigated" && isMainFrame(params)) {
        this.refStore.clear(session.tabId)
      }
    })
  }

  async take(clientId: string, request: SnapshotRequest): Promise<SnapshotResult> {
    const session = await this.sessionManager.require(request.tabId)
    const target = this.sessionManager.target(request.tabId)
    const nodes = request.selector
      ? await this.readSelectorTree(target, request.selector)
      : (await this.cdpClient.send<AxTreeResponse>(target, "Accessibility.getFullAXTree")).nodes
    const normalized = normalizeAxTree(nodes)
      .filter((node) => !request.interactiveOnly || node.interactive)
      .filter((node) => request.maxDepth === undefined || node.depth <= request.maxDepth)
    const snapshotId = `snap_${crypto.randomUUID()}`
    const refs = this.createRefs(clientId, request.tabId, session.documentEpoch, snapshotId, normalized)
    const formatted = this.formatter.format(
      request.compact ? compactNodes(refs.decorated) : refs.decorated,
      request.includeUrls ?? false,
      request.maxChars ?? 100_000
    )
    const tab = await chrome.tabs.get(request.tabId)
    const origin = readOrigin(tab.url)
    this.refStore.replace({
      snapshotId,
      tabId: request.tabId,
      documentEpoch: session.documentEpoch,
      refs: refs.internal
    })
    session.latestSnapshotId = snapshotId

    return {
      snapshotId,
      tabId: request.tabId,
      documentEpoch: session.documentEpoch,
      origin,
      text: formatted.text,
      refs: refs.public,
      truncated: formatted.truncated
    }
  }

  private createRefs(
    clientId: string,
    tabId: number,
    documentEpoch: number,
    snapshotId: string,
    nodes: NormalizedAxNode[]
  ): SnapshotRefs {
    const allocator = new RefAllocator()
    const duplicates = new Map<string, number>()
    const internal = new Map<string, InternalRefEntry>()
    const publicRefs: Record<string, PublicRef> = {}
    const decorated = nodes.map((node): DecoratedNode => {
      if (!node.interactive || node.backendNodeId === undefined) {
        return node
      }
      const duplicateKey = `${node.role}\u0000${node.name}`
      const nth = duplicates.get(duplicateKey) ?? 0
      duplicates.set(duplicateKey, nth + 1)
      const refId = allocator.next()
      const publicRef: PublicRef = {
        role: node.role,
        name: node.name,
        ...(nth > 0 ? { nth } : {})
      }
      internal.set(refId, {
        ...publicRef,
        refId,
        snapshotId,
        clientId,
        tabId,
        documentEpoch,
        backendNodeId: node.backendNodeId,
        createdAt: Date.now()
      })
      publicRefs[refId] = publicRef
      return { ...node, refId }
    })
    return { decorated, internal, public: publicRefs }
  }

  private async readSelectorTree(target: { tabId: number }, selector: string): Promise<AxNode[]> {
    const evaluated = await this.cdpClient.send<RuntimeEvaluateResponse>(target, "Runtime.evaluate", {
      expression: `document.querySelector(${JSON.stringify(selector)})`,
      returnByValue: false
    })
    const objectId = evaluated.result.objectId
    if (!objectId || evaluated.result.subtype === "null") {
      throw new BrowserProtocolError("UNKNOWN_REF", "Snapshot selector did not match an element")
    }
    const described = await this.cdpClient.send<DomDescribeResponse>(target, "DOM.describeNode", { objectId })
    if (!described.node.backendNodeId) {
      throw new BrowserProtocolError("UNKNOWN_REF", "Snapshot selector could not be resolved")
    }
    const response = await this.cdpClient.send<AxTreeResponse>(target, "Accessibility.getPartialAXTree", {
      backendNodeId: described.node.backendNodeId,
      fetchRelatives: true
    })
    return response.nodes
  }
}

function compactNodes<T extends { role: string; name: string }>(nodes: T[]): T[] {
  return nodes.filter((node, index) => {
    const previous = nodes[index - 1]
    return !previous || previous.role !== node.role || previous.name !== node.name
  })
}

function readOrigin(url: string | undefined): string {
  if (!url) {
    return ""
  }
  try {
    return new URL(url).origin
  } catch {
    return ""
  }
}

function isMainFrame(params: unknown): boolean {
  return typeof params === "object" && params !== null && "frame" in params
}
