export const browserMethods = [
  "tabs.list",
  "tabs.getActive",
  "page.snapshot",
  "page.getMetadata",
  "page.getText",
  "page.getHtml",
  "page.navigate",
  "page.reload",
  "element.click",
  "element.fill",
  "element.type",
  "element.getText",
  "network.startCapture",
  "network.listRequests",
  "network.getResponseBody",
  "debug.getConsole",
  "debug.getErrors",
  "debug.evaluate",
  "policy.listAllowedOrigins"
] as const

export type BrowserMethod = (typeof browserMethods)[number]
