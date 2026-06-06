export type CdpTarget = {
  tabId: number
  sessionId?: string
}

export class CdpClient {
  send<TResult>(
    target: CdpTarget,
    method: string,
    params?: Record<string, unknown>
  ): Promise<TResult> {
    return chrome.debugger.sendCommand(target, method, params) as Promise<TResult>
  }
}
