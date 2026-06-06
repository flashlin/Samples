import { describe, expect, test } from "bun:test"
import { RefAllocator } from "../src/refs/ref-allocator"
import { RefStore, type InternalRefEntry, type StoredSnapshot } from "../src/refs/ref-store"
import {
  normalizeAxTree,
  type AxNode,
  type NormalizedAxNode
} from "../src/snapshots/ax-tree-normalizer"
import { SnapshotFormatter } from "../src/snapshots/snapshot-formatter"

describe("normalizeAxTree", () => {
  test("removes ignored and non-semantic nodes", () => {
    const nodes: AxNode[] = [
      { nodeId: "ignored", ignored: true, role: { value: "button" }, name: { value: "Ignored" } },
      { nodeId: "empty", role: { value: "" }, name: { value: "Empty" } },
      { nodeId: "none", role: { value: "none" }, name: { value: "None" } },
      { nodeId: "generic", role: { value: "generic" }, name: { value: "Generic" } },
      { nodeId: "heading", role: { value: "heading" }, name: { value: "Visible" } }
    ]

    expect(normalizeAxTree(nodes)).toEqual([
      {
        nodeId: "heading",
        role: "heading",
        name: "Visible",
        depth: 0,
        interactive: false
      }
    ])
  })

  test("calculates depth through the parent chain", () => {
    const nodes: AxNode[] = [
      { nodeId: "root", role: { value: "RootWebArea" } },
      { nodeId: "group", parentId: "root", role: { value: "group" } },
      { nodeId: "button", parentId: "group", role: { value: "button" }, name: { value: "Submit" } }
    ]

    expect(normalizeAxTree(nodes).map(({ nodeId, depth }) => ({ nodeId, depth }))).toEqual([
      { nodeId: "root", depth: 0 },
      { nodeId: "group", depth: 1 },
      { nodeId: "button", depth: 2 }
    ])
  })

  test("identifies interactive roles", () => {
    const nodes: AxNode[] = [
      { nodeId: "link", role: { value: "link" }, name: { value: "Details" } },
      { nodeId: "textbox", role: { value: "textbox" }, name: { value: "Email" } },
      { nodeId: "heading", role: { value: "heading" }, name: { value: "Title" } }
    ]

    expect(normalizeAxTree(nodes).map(({ nodeId, interactive }) => ({ nodeId, interactive }))).toEqual([
      { nodeId: "link", interactive: true },
      { nodeId: "textbox", interactive: true },
      { nodeId: "heading", interactive: false }
    ])
  })

  test("normalizes backend node IDs and string URL properties", () => {
    const nodes: AxNode[] = [
      {
        nodeId: "link",
        backendDOMNodeId: 42,
        role: { value: "link" },
        name: { value: 123 },
        properties: [{ name: "url", value: { value: "https://example.com/path?q=1" } }]
      },
      {
        nodeId: "button",
        role: { value: "button" },
        properties: [{ name: "url", value: { value: 123 } }]
      }
    ]

    expect(normalizeAxTree(nodes)).toEqual([
      {
        nodeId: "link",
        backendNodeId: 42,
        role: "link",
        name: "123",
        depth: 0,
        interactive: true,
        url: "https://example.com/path?q=1"
      },
      {
        nodeId: "button",
        role: "button",
        name: "",
        depth: 0,
        interactive: true
      }
    ])
  })
})

describe("SnapshotFormatter", () => {
  test("formats indentation, refs, and optional URLs", () => {
    const formatter = new SnapshotFormatter()
    const nodes: Array<NormalizedAxNode & { refId?: string }> = [
      {
        nodeId: "heading",
        role: "heading",
        name: "Title",
        depth: 0,
        interactive: false
      },
      {
        nodeId: "link",
        backendNodeId: 42,
        role: "link",
        name: "Details",
        depth: 1,
        interactive: true,
        url: "https://example.com/details",
        refId: "e1"
      }
    ]

    expect(formatter.format(nodes, true, 1_000)).toEqual({
      text: [
        '- heading "Title"',
        '  - link "Details" [ref=e1] [url=https://example.com/details]'
      ].join("\n"),
      truncated: false
    })
    expect(formatter.format(nodes, false, 1_000).text).not.toContain("[url=")
  })

  test("escapes backslashes, quotes, and newlines", () => {
    const formatter = new SnapshotFormatter()
    const nodes: NormalizedAxNode[] = [
      {
        nodeId: "textbox",
        role: "textbox",
        name: "Line one\nLine \"two\" \\ end",
        depth: 0,
        interactive: true
      }
    ]

    expect(formatter.format(nodes, false, 1_000)).toEqual({
      text: '- textbox "Line one Line \\"two\\" \\\\ end"',
      truncated: false
    })
  })

  test("truncates output at the maximum character count", () => {
    const formatter = new SnapshotFormatter()
    const nodes: NormalizedAxNode[] = [
      {
        nodeId: "heading",
        role: "heading",
        name: "Long title",
        depth: 0,
        interactive: false
      }
    ]

    expect(formatter.format(nodes, false, 10)).toEqual({
      text: "- heading ",
      truncated: true
    })
  })
})

describe("RefAllocator", () => {
  test("allocates sequential ref IDs from e1", () => {
    const allocator = new RefAllocator()

    expect([allocator.next(), allocator.next(), allocator.next()]).toEqual(["e1", "e2", "e3"])
  })
})

describe("RefStore", () => {
  test("replaces the latest snapshot for a tab", () => {
    const store = new RefStore()
    const first = createSnapshot(7, "snapshot-1", "e1")
    const latest = createSnapshot(7, "snapshot-2", "e2")

    store.replace(first)
    store.replace(latest)

    expect(store.get(7, "snapshot-1")).toBeUndefined()
    expect(store.get(7, "snapshot-2")).toBe(latest)
    expect(store.snapshots.size).toBe(1)
  })
})

function createSnapshot(tabId: number, snapshotId: string, refId: string): StoredSnapshot {
  const entry: InternalRefEntry = {
    refId,
    snapshotId,
    clientId: "chrome-default",
    tabId,
    documentEpoch: 1,
    backendNodeId: 42,
    role: "button",
    name: "Submit",
    createdAt: 1
  }

  return {
    snapshotId,
    tabId,
    documentEpoch: 1,
    refs: new Map([[refId, entry]])
  }
}
