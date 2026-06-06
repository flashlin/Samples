export type AxValue = {
  value?: unknown
}

export type AxNode = {
  nodeId: string
  backendDOMNodeId?: number
  ignored?: boolean
  role?: AxValue
  name?: AxValue
  properties?: Array<{ name: string; value?: AxValue }>
  childIds?: string[]
  parentId?: string
}

export type NormalizedAxNode = {
  nodeId: string
  backendNodeId?: number
  role: string
  name: string
  depth: number
  interactive: boolean
  url?: string
}

const interactiveRoles = new Set([
  "button",
  "checkbox",
  "combobox",
  "link",
  "menuitem",
  "option",
  "radio",
  "searchbox",
  "slider",
  "spinbutton",
  "switch",
  "tab",
  "textbox"
])

export function normalizeAxTree(nodes: AxNode[]): NormalizedAxNode[] {
  const nodesById = new Map(nodes.map((node) => [node.nodeId, node]))
  return nodes
    .filter((node) => !node.ignored)
    .map((node): NormalizedAxNode => {
      const role = readValue(node.role)
      const name = readValue(node.name)
      const url = readProperty(node, "url")
      return {
        nodeId: node.nodeId,
        ...(node.backendDOMNodeId === undefined ? {} : { backendNodeId: node.backendDOMNodeId }),
        role,
        name,
        depth: readDepth(node, nodesById),
        interactive: interactiveRoles.has(role),
        ...(url === undefined ? {} : { url })
      }
    })
    .filter((node) => node.role && node.role !== "none" && node.role !== "generic")
}

function readDepth(node: AxNode, nodesById: Map<string, AxNode>): number {
  let depth = 0
  let parentId = node.parentId
  while (parentId) {
    depth += 1
    parentId = nodesById.get(parentId)?.parentId
  }
  return depth
}

function readProperty(node: AxNode, propertyName: string): string | undefined {
  const property = node.properties?.find(({ name }) => name === propertyName)
  const value = property?.value?.value
  return typeof value === "string" ? value : undefined
}

function readValue(value: AxValue | undefined): string {
  const raw = value?.value
  return typeof raw === "string" ? raw : raw === undefined ? "" : String(raw)
}
