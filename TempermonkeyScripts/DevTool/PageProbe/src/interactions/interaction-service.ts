import { BrowserProtocolError, type RefTarget } from "@page-probe/protocol"
import type { CdpClient } from "../cdp/cdp-client"
import type { TabSessionManager } from "../cdp/tab-session-manager"
import type { RefResolver, ResolvedElement } from "../refs/ref-resolver"

type BoxModelResponse = {
  model: { content: number[]; border: number[] }
}

export class InteractionService {
  constructor(
    private readonly cdpClient: CdpClient,
    private readonly sessionManager: TabSessionManager,
    private readonly refResolver: RefResolver
  ) {}

  async click(target: RefTarget): Promise<{ success: true }> {
    const resolved = await this.refResolver.resolve(target)
    const point = await this.readCenter(resolved)
    const cdpTarget = this.sessionManager.target(target.tabId, resolved.entry.sessionId)
    await this.cdpClient.send(cdpTarget, "Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: point.x,
      y: point.y
    })
    await this.cdpClient.send(cdpTarget, "Input.dispatchMouseEvent", {
      type: "mousePressed",
      x: point.x,
      y: point.y,
      button: "left",
      clickCount: 1
    })
    await this.cdpClient.send(cdpTarget, "Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: point.x,
      y: point.y,
      button: "left",
      clickCount: 1
    })
    return { success: true }
  }

  async fill(target: RefTarget, value: string): Promise<{ success: true }> {
    const resolved = await this.refResolver.resolve(target)
    await this.callFunction(
      resolved,
      `function(text) {
        this.focus();
        this.value = text;
        this.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
        this.dispatchEvent(new Event("change", { bubbles: true }));
      }`,
      value
    )
    return { success: true }
  }

  async type(target: RefTarget, text: string): Promise<{ success: true }> {
    const resolved = await this.refResolver.resolve(target)
    await this.callFunction(resolved, "function() { this.focus(); }")
    await this.cdpClient.send(
      this.sessionManager.target(target.tabId, resolved.entry.sessionId),
      "Input.insertText",
      { text }
    )
    return { success: true }
  }

  async getText(target: RefTarget): Promise<{ text: string; truncated: boolean }> {
    const resolved = await this.refResolver.resolve(target)
    const response = await this.cdpClient.send<{ result: { value?: unknown } }>(
      this.sessionManager.target(target.tabId, resolved.entry.sessionId),
      "Runtime.callFunctionOn",
      {
        objectId: resolved.objectId,
        functionDeclaration: "function() { return this.innerText ?? this.textContent ?? ''; }",
        returnByValue: true
      }
    )
    return {
      text: typeof response.result.value === "string" ? response.result.value : "",
      truncated: false
    }
  }

  private async readCenter(resolved: ResolvedElement): Promise<{ x: number; y: number }> {
    let response: BoxModelResponse
    try {
      response = await this.cdpClient.send<BoxModelResponse>(
        this.sessionManager.target(resolved.entry.tabId, resolved.entry.sessionId),
        "DOM.getBoxModel",
        { backendNodeId: resolved.backendNodeId }
      )
    } catch {
      throw new BrowserProtocolError("ELEMENT_NOT_VISIBLE", "Element does not have a visible box")
    }
    const quad = response.model.border.length >= 8 ? response.model.border : response.model.content
    return {
      x: (quad[0]! + quad[2]! + quad[4]! + quad[6]!) / 4,
      y: (quad[1]! + quad[3]! + quad[5]! + quad[7]!) / 4
    }
  }

  private async callFunction(
    resolved: ResolvedElement,
    functionDeclaration: string,
    ...arguments_: string[]
  ): Promise<void> {
    await this.cdpClient.send(
      this.sessionManager.target(resolved.entry.tabId, resolved.entry.sessionId),
      "Runtime.callFunctionOn",
      {
        objectId: resolved.objectId,
        functionDeclaration,
        arguments: arguments_.map((value) => ({ value })),
        awaitPromise: true
      }
    )
  }
}
