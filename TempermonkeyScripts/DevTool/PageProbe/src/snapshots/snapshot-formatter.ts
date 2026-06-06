import type { NormalizedAxNode } from "./ax-tree-normalizer"

export type FormattedSnapshot = {
  text: string
  truncated: boolean
}

export class SnapshotFormatter {
  format(
    nodes: Array<NormalizedAxNode & { refId?: string }>,
    includeUrls: boolean,
    maxChars: number
  ): FormattedSnapshot {
    const text = nodes.map((node) => formatNode(node, includeUrls)).join("\n")
    if (text.length <= maxChars) {
      return { text, truncated: false }
    }
    return { text: text.slice(0, maxChars), truncated: true }
  }
}

function formatNode(node: NormalizedAxNode & { refId?: string }, includeUrls: boolean): string {
  const indentation = "  ".repeat(node.depth)
  const name = node.name ? ` "${escapeText(node.name)}"` : ""
  const ref = node.refId ? ` [ref=${node.refId}]` : ""
  const url = includeUrls && node.url ? ` [url=${node.url}]` : ""
  return `${indentation}- ${node.role}${name}${ref}${url}`
}

function escapeText(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("\"", "\\\"").replaceAll("\n", " ")
}
