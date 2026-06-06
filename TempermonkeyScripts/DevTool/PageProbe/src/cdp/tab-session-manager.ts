import { BrowserProtocolError } from "@page-probe/protocol"
import type { CdpClient, CdpTarget } from "./cdp-client"

export type TabSession = {
  tabId: number
  attached: boolean
  documentEpoch: number
  childSessions: Map<string, string>
  latestSnapshotId: string | undefined
}

type CdpEventListener = (
  session: TabSession,
  source: chrome.debugger.Debuggee,
  method: string,
  params: unknown
) => void

export class TabSessionManager {
  readonly sessions = new Map<number, TabSession>()
  readonly eventListeners = new Set<CdpEventListener>()

  constructor(private readonly cdpClient: CdpClient) {
    chrome.debugger.onEvent.addListener((source, method, params) => {
      this.handleEvent(source, method, params)
    })
    chrome.debugger.onDetach.addListener((source) => {
      if (source.tabId) {
        this.invalidate(source.tabId)
      }
    })
    chrome.tabs.onRemoved.addListener((tabId) => this.sessions.delete(tabId))
  }

  subscribe(listener: CdpEventListener): () => void {
    this.eventListeners.add(listener)
    return () => this.eventListeners.delete(listener)
  }

  async require(tabId: number): Promise<TabSession> {
    const existing = this.sessions.get(tabId)
    if (existing?.attached) {
      return existing
    }

    const session = existing ?? {
      tabId,
      attached: false,
      documentEpoch: 0,
      childSessions: new Map<string, string>(),
      latestSnapshotId: undefined
    }

    try {
      await chrome.debugger.attach({ tabId }, "1.3")
      session.attached = true
      this.sessions.set(tabId, session)
      await this.enableDomains(tabId)
      return session
    } catch (error) {
      throw new BrowserProtocolError(
        "DEBUGGER_ATTACH_FAILED",
        "Unable to attach Chrome debugger",
        error instanceof Error ? error.message : error
      )
    }
  }

  target(tabId: number, sessionId?: string): CdpTarget {
    return sessionId ? { tabId, sessionId } : { tabId }
  }

  invalidate(tabId: number): void {
    const session = this.sessions.get(tabId)
    if (!session) {
      return
    }
    session.attached = false
    session.documentEpoch += 1
    session.latestSnapshotId = undefined
    session.childSessions.clear()
  }

  private async enableDomains(tabId: number): Promise<void> {
    const target = { tabId }
    await Promise.all([
      this.cdpClient.send(target, "Page.enable"),
      this.cdpClient.send(target, "DOM.enable"),
      this.cdpClient.send(target, "Accessibility.enable"),
      this.cdpClient.send(target, "Runtime.enable"),
      this.cdpClient.send(target, "Log.enable"),
      this.cdpClient.send(target, "Network.enable"),
      this.cdpClient.send(target, "Target.setAutoAttach", {
        autoAttach: true,
        waitForDebuggerOnStart: false,
        flatten: true
      })
    ])
  }

  private handleEvent(source: chrome.debugger.Debuggee, method: string, params: unknown): void {
    const tabId = source.tabId
    if (!tabId) {
      return
    }

    const session = this.sessions.get(tabId)
    if (!session) {
      return
    }

    if (method === "Page.frameNavigated" && isMainFrameNavigation(params)) {
      session.documentEpoch += 1
      session.latestSnapshotId = undefined
    }

    if (method === "Target.attachedToTarget") {
      const attached = readAttachedTarget(params)
      if (attached) {
        session.childSessions.set(attached.targetId, attached.sessionId)
      }
    }

    if (method === "Target.detachedFromTarget") {
      const sessionId = readDetachedSessionId(params)
      if (sessionId) {
        for (const [targetId, childSessionId] of session.childSessions) {
          if (childSessionId === sessionId) {
            session.childSessions.delete(targetId)
          }
        }
      }
    }

    for (const listener of this.eventListeners) {
      listener(session, source, method, params)
    }
  }
}

function isMainFrameNavigation(params: unknown): boolean {
  if (!isRecord(params) || !isRecord(params.frame)) {
    return false
  }
  return !("parentId" in params.frame)
}

function readAttachedTarget(params: unknown): { targetId: string; sessionId: string } | undefined {
  if (!isRecord(params) || typeof params.sessionId !== "string" || !isRecord(params.targetInfo)) {
    return undefined
  }
  return typeof params.targetInfo.targetId === "string"
    ? { targetId: params.targetInfo.targetId, sessionId: params.sessionId }
    : undefined
}

function readDetachedSessionId(params: unknown): string | undefined {
  return isRecord(params) && typeof params.sessionId === "string" ? params.sessionId : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}
