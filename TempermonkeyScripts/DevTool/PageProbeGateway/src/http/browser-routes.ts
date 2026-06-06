import {
  debugEvaluateParamsSchema,
  debugGetConsoleParamsSchema,
  elementClickParamsSchema,
  elementFillParamsSchema,
  elementGetTextParamsSchema,
  elementTypeParamsSchema,
  networkGetResponseBodyParamsSchema,
  networkListRequestsParamsSchema,
  networkStartCaptureParamsSchema,
  pageHtmlParamsSchema,
  pageTextParamsSchema,
  snapshotRequestSchema,
  tabIdSchema,
  type BrowserMethod
} from "@page-probe/protocol"
import type { BrowserToolService } from "../application/browser-tool-service"

type RouteMatch = {
  method: BrowserMethod
  params: unknown
}

export async function handleBrowserRoute(
  request: Request,
  pathname: string,
  service: BrowserToolService
): Promise<Response | undefined> {
  const match = await matchBrowserRoute(request, pathname)
  if (!match) {
    return undefined
  }

  const clientId = request.headers.get("x-page-probe-client") ?? undefined
  const result = await service.execute(match.method, match.params as never, clientId)
  return Response.json(result)
}

async function matchBrowserRoute(request: Request, pathname: string): Promise<RouteMatch | undefined> {
  if (request.method === "GET" && pathname === "/api/tabs") {
    return { method: "tabs.list", params: {} }
  }

  const snapshot = matchTabRoute(pathname, "/snapshot")
  if (request.method === "POST" && snapshot) {
    return {
      method: "page.snapshot",
      params: snapshotRequestSchema.parse({ tabId: snapshot.tabId, ...await readJson(request) })
    }
  }

  const html = matchTabRoute(pathname, "/html")
  if (request.method === "GET" && html) {
    return { method: "page.getHtml", params: pageHtmlParamsSchema.parse({ tabId: html.tabId }) }
  }

  const text = matchTabRoute(pathname, "/text")
  if (request.method === "GET" && text) {
    return { method: "page.getText", params: pageTextParamsSchema.parse({ tabId: text.tabId }) }
  }

  const network = matchTabRoute(pathname, "/network")
  if (request.method === "POST" && network) {
    return {
      method: "network.startCapture",
      params: networkStartCaptureParamsSchema.parse({ tabId: network.tabId })
    }
  }
  if (request.method === "GET" && network) {
    return {
      method: "network.listRequests",
      params: networkListRequestsParamsSchema.parse({ tabId: network.tabId })
    }
  }

  const elementMatch = pathname.match(/^\/api\/tabs\/(\d+)\/elements\/([^/]+)\/(click|fill|type|text)$/)
  if (elementMatch) {
    const tabId = tabIdSchema.parse(Number(elementMatch[1]))
    const ref = decodeURIComponent(elementMatch[2] ?? "")
    const action = elementMatch[3]
    const body = request.method === "POST" ? await readJson(request) : {}
    const base = { tabId, snapshotId: readSnapshotId(request, body), ref }

    if (request.method === "POST" && action === "click") {
      return { method: "element.click", params: elementClickParamsSchema.parse(base) }
    }
    if (request.method === "POST" && action === "fill") {
      return { method: "element.fill", params: elementFillParamsSchema.parse({ ...base, ...body }) }
    }
    if (request.method === "POST" && action === "type") {
      return { method: "element.type", params: elementTypeParamsSchema.parse({ ...base, ...body }) }
    }
    if (request.method === "GET" && action === "text") {
      return { method: "element.getText", params: elementGetTextParamsSchema.parse(base) }
    }
  }

  const bodyMatch = pathname.match(/^\/api\/tabs\/(\d+)\/network\/([^/]+)\/body$/)
  if (request.method === "GET" && bodyMatch) {
    return {
      method: "network.getResponseBody",
      params: networkGetResponseBodyParamsSchema.parse({
        tabId: Number(bodyMatch[1]),
        requestId: decodeURIComponent(bodyMatch[2] ?? "")
      })
    }
  }

  const consoleMatch = matchTabRoute(pathname, "/debug/console")
  if (request.method === "GET" && consoleMatch) {
    return {
      method: "debug.getConsole",
      params: debugGetConsoleParamsSchema.parse({ tabId: consoleMatch.tabId })
    }
  }

  const errorsMatch = matchTabRoute(pathname, "/debug/errors")
  if (request.method === "GET" && errorsMatch) {
    return {
      method: "debug.getErrors",
      params: debugGetConsoleParamsSchema.parse({ tabId: errorsMatch.tabId })
    }
  }

  const evaluateMatch = matchTabRoute(pathname, "/debug/evaluate")
  if (request.method === "POST" && evaluateMatch) {
    return {
      method: "debug.evaluate",
      params: debugEvaluateParamsSchema.parse({ tabId: evaluateMatch.tabId, ...await readJson(request) })
    }
  }

  return undefined
}

function matchTabRoute(pathname: string, suffix: string): { tabId: number } | undefined {
  const match = pathname.match(new RegExp(`^/api/tabs/(\\d+)${suffix}$`))
  return match ? { tabId: tabIdSchema.parse(Number(match[1])) } : undefined
}

async function readJson(request: Request): Promise<Record<string, unknown>> {
  const body = await request.json()
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return {}
  }
  return body as Record<string, unknown>
}

function readSnapshotId(request: Request, body: Record<string, unknown>): unknown {
  return body.snapshotId ?? request.headers.get("x-page-probe-snapshot")
}
