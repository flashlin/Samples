import type { BrowserMethod } from "@page-probe/protocol"

const longRunningMethods = new Set<BrowserMethod>([
  "page.snapshot",
  "network.getResponseBody"
])

export class TimeoutPolicy {
  readonly defaultTimeoutMs: number

  constructor(defaultTimeoutMs: number) {
    this.defaultTimeoutMs = defaultTimeoutMs
  }

  for(method: BrowserMethod): number {
    return longRunningMethods.has(method) ? Math.min(this.defaultTimeoutMs * 4, 60_000) : this.defaultTimeoutMs
  }
}
