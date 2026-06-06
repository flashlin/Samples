import { BrowserProtocolError } from "@page-probe/protocol"

export class RequestOriginGuard {
  readonly allowedOrigins: Set<string>
  readonly allowedHosts: Set<string>

  constructor(allowedOrigins: string[], allowedHosts: string[]) {
    this.allowedOrigins = new Set(allowedOrigins)
    this.allowedHosts = new Set(allowedHosts)
  }

  assertRequest(request: Request): void {
    this.assertHost(request)

    const origin = request.headers.get("origin")
    if (origin && !this.allowedOrigins.has(origin)) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Request origin is not allowed")
    }
  }

  assertExtensionRequest(request: Request): void {
    this.assertHost(request)
    const origin = request.headers.get("origin")
    if (!origin?.startsWith("chrome-extension://")) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Extension origin is not allowed")
    }
  }

  private assertHost(request: Request): void {
    const host = request.headers.get("host")
    if (!host || !this.allowedHosts.has(host)) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Request host is not allowed")
    }
  }
}
