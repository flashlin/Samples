import type { CdpClient } from "../cdp/cdp-client"
import type { TabSessionManager } from "../cdp/tab-session-manager"

type EvaluateResponse = { result: { value?: unknown } }

export class PageService {
  constructor(
    private readonly cdpClient: CdpClient,
    private readonly sessionManager: TabSessionManager
  ) {}

  async getMetadata(tabId: number) {
    const tab = await chrome.tabs.get(tabId)
    return {
      tabId,
      title: tab.title ?? "",
      url: tab.url ?? "",
      origin: readOrigin(tab.url)
    }
  }

  async getText(tabId: number, selector?: string, maxChars = 100_000) {
    const expression = selector
      ? `(document.querySelector(${JSON.stringify(selector)})?.innerText ?? "")`
      : `(document.body?.innerText ?? "")`
    const text = await this.evaluateString(tabId, expression)
    return {
      text: text.slice(0, maxChars),
      truncated: text.length > maxChars
    }
  }

  async getHtml(tabId: number, selector?: string, maxBytes = 1_048_576, sanitize = true) {
    const expression = createHtmlExpression(selector, sanitize)
    const html = await this.evaluateString(tabId, expression)
    const encoded = new TextEncoder().encode(html)
    if (encoded.byteLength <= maxBytes) {
      return { html, truncated: false }
    }
    return { html: html.slice(0, maxBytes), truncated: true }
  }

  async navigate(tabId: number, url: string) {
    await this.sessionManager.require(tabId)
    await this.cdpClient.send(this.sessionManager.target(tabId), "Page.navigate", { url })
    return { success: true as const }
  }

  async reload(tabId: number, ignoreCache?: boolean) {
    await this.sessionManager.require(tabId)
    const params = ignoreCache === undefined ? {} : { ignoreCache }
    await this.cdpClient.send(this.sessionManager.target(tabId), "Page.reload", params)
    return { success: true as const }
  }

  async goBack(tabId: number) {
    return this.runHistoryNavigation(tabId, "history.back()")
  }

  async goForward(tabId: number) {
    return this.runHistoryNavigation(tabId, "history.forward()")
  }

  private async runHistoryNavigation(tabId: number, expression: string) {
    await this.sessionManager.require(tabId)
    await this.cdpClient.send(this.sessionManager.target(tabId), "Runtime.evaluate", { expression })
    return { success: true as const }
  }

  private async evaluateString(tabId: number, expression: string): Promise<string> {
    await this.sessionManager.require(tabId)
    const response = await this.cdpClient.send<EvaluateResponse>(
      this.sessionManager.target(tabId),
      "Runtime.evaluate",
      {
        expression,
        returnByValue: true
      }
    )
    return typeof response.result.value === "string" ? response.result.value : ""
  }
}

function createHtmlExpression(selector: string | undefined, sanitize: boolean): string {
  const rootExpression = selector
    ? `document.querySelector(${JSON.stringify(selector)})`
    : "document.documentElement"
  if (!sanitize) {
    return `(${rootExpression}?.outerHTML ?? "")`
  }
  return `(() => {
    const source = ${rootExpression};
    if (!source) return "";
    const clone = source.cloneNode(true);
    clone.querySelectorAll("script, style, link[rel=stylesheet]").forEach(node => node.remove());
    clone.querySelectorAll("*").forEach(node => {
      for (const attribute of [...node.attributes]) {
        if (attribute.name.startsWith("on")) node.removeAttribute(attribute.name);
      }
    });
    return clone.outerHTML;
  })()`
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
