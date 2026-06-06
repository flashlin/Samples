import { timingSafeEqual } from "node:crypto"
import { BrowserProtocolError } from "@page-probe/protocol"

export class BearerAuth {
  readonly token: string

  constructor(token: string) {
    this.token = token
  }

  assertRequest(request: Request): void {
    const authorization = request.headers.get("authorization")
    if (!authorization?.startsWith("Bearer ")) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Bearer token is required")
    }

    const candidate = authorization.slice("Bearer ".length)
    if (!tokensMatch(candidate, this.token)) {
      throw new BrowserProtocolError("PERMISSION_DENIED", "Bearer token is invalid")
    }
  }

  matches(candidate: string): boolean {
    return tokensMatch(candidate, this.token)
  }
}

function tokensMatch(candidate: string, expected: string): boolean {
  const candidateBytes = Buffer.from(candidate)
  const expectedBytes = Buffer.from(expected)
  return candidateBytes.length === expectedBytes.length && timingSafeEqual(candidateBytes, expectedBytes)
}
