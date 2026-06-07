import type { BrowserCommandParams } from "@page-probe/protocol"
import type { DebugEventService } from "../debug/debug-event-service"
import type { InteractionService } from "../interactions/interaction-service"
import type { NetworkCaptureService } from "../network/network-capture-service"
import type { PageService } from "../page/page-service"
import type { PolicyService } from "../security/policy-service"
import type { SnapshotService } from "../snapshots/snapshot-service"
import type { TabService } from "../tabs/tab-service"
import type { HandlerRegistry } from "./handler-registry"

export type HandlerServices = {
  tabs: TabService
  page: PageService
  snapshot: SnapshotService
  interaction: InteractionService
  network: NetworkCaptureService
  debug: DebugEventService
  policy: PolicyService
}

export function registerHandlers(registry: HandlerRegistry, services: HandlerServices): void {
  registry.register("tabs.list", () => services.tabs.list())
  registry.register("tabs.getActive", () => services.tabs.getActive())
  registry.register("page.snapshot", (params, clientId) => {
    return services.snapshot.take(clientId, params as BrowserCommandParams["page.snapshot"])
  })
  registry.register("page.getMetadata", (params) => {
    const input = params as BrowserCommandParams["page.getMetadata"]
    return services.page.getMetadata(input.tabId)
  })
  registry.register("page.getText", (params) => {
    const input = params as BrowserCommandParams["page.getText"]
    return services.page.getText(input.tabId, input.selector, input.maxChars)
  })
  registry.register("page.getHtml", (params) => {
    const input = params as BrowserCommandParams["page.getHtml"]
    return services.page.getHtml(input.tabId, input.selector, input.maxBytes, input.sanitize)
  })
  registry.register("page.navigate", (params) => {
    const input = params as BrowserCommandParams["page.navigate"]
    return services.page.navigate(input.tabId, input.url)
  })
  registry.register("page.reload", (params) => {
    const input = params as BrowserCommandParams["page.reload"]
    return services.page.reload(input.tabId, input.ignoreCache)
  })
  registry.register("element.click", (params) => {
    return services.interaction.click(params as BrowserCommandParams["element.click"])
  })
  registry.register("element.fill", (params) => {
    const input = params as BrowserCommandParams["element.fill"]
    return services.interaction.fill(input, input.value)
  })
  registry.register("element.type", (params) => {
    const input = params as BrowserCommandParams["element.type"]
    return services.interaction.type(input, input.text)
  })
  registry.register("element.getText", (params) => {
    return services.interaction.getText(params as BrowserCommandParams["element.getText"])
  })
  registry.register("network.startCapture", (params) => {
    const input = params as BrowserCommandParams["network.startCapture"]
    return services.network.start(input.tabId)
  })
  registry.register("network.listRequests", (params) => {
    const input = params as BrowserCommandParams["network.listRequests"]
    return services.network.list(input.tabId, input.limit)
  })
  registry.register("network.getResponseBody", (params) => {
    const input = params as BrowserCommandParams["network.getResponseBody"]
    return services.network.getResponseBody(input.tabId, input.requestId)
  })
  registry.register("debug.getConsole", (params) => {
    const input = params as BrowserCommandParams["debug.getConsole"]
    return services.debug.getConsole(input.tabId, input.limit)
  })
  registry.register("debug.getErrors", (params) => {
    const input = params as BrowserCommandParams["debug.getErrors"]
    return services.debug.getErrors(input.tabId, input.limit)
  })
  registry.register("debug.evaluate", (params) => {
    const input = params as BrowserCommandParams["debug.evaluate"]
    return services.debug.evaluate(input.tabId, input.expression, input.returnByValue)
  })
  registry.register("policy.listAllowedOrigins", () => services.policy.listAllowedOrigins())
}
