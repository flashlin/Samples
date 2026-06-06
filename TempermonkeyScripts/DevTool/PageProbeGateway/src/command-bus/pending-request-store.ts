import {
  BrowserProtocolError,
  type BrowserMethod,
  type BrowserCommandResults,
  type ErrorEnvelope,
  type ResultEnvelope,
  parseResultEnvelope
} from "@page-probe/protocol"

type PendingRequest<TMethod extends BrowserMethod = BrowserMethod> = {
  clientId: string
  method: TMethod
  timeout: ReturnType<typeof setTimeout>
  resolve: (value: BrowserCommandResults[TMethod]) => void
  reject: (reason: Error) => void
}

export class PendingRequestStore {
  readonly requests = new Map<string, PendingRequest>()

  create<TMethod extends BrowserMethod>(
    requestId: string,
    clientId: string,
    method: TMethod,
    timeoutMs: number
  ): Promise<BrowserCommandResults[TMethod]> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.requests.delete(requestId)
        reject(new BrowserProtocolError("ACTION_TIMEOUT", `Command ${method} timed out`))
      }, timeoutMs)

      this.requests.set(requestId, {
        clientId,
        method,
        timeout,
        resolve: resolve as PendingRequest["resolve"],
        reject
      })
    })
  }

  settle(envelope: ResultEnvelope | ErrorEnvelope): void {
    const pending = this.requests.get(envelope.id)
    if (!pending) {
      return
    }

    clearTimeout(pending.timeout)
    this.requests.delete(envelope.id)
    if (envelope.ok) {
      pending.resolve(envelope.result)
      return
    }

    pending.reject(new BrowserProtocolError(envelope.error.code, envelope.error.message, envelope.error.details))
  }

  settleUnknown(input: unknown): void {
    const id = readResultId(input)
    const pending = this.requests.get(id)
    if (!pending) {
      return
    }

    this.settle(parseResultEnvelope(pending.method, input))
  }

  rejectClient(clientId: string): void {
    for (const [requestId, pending] of this.requests) {
      if (pending.clientId !== clientId) {
        continue
      }

      clearTimeout(pending.timeout)
      this.requests.delete(requestId)
      pending.reject(new BrowserProtocolError("CLIENT_NOT_CONNECTED", "Browser client disconnected"))
    }
  }
}

function readResultId(input: unknown): string {
  if (typeof input !== "object" || input === null || !("id" in input) || typeof input.id !== "string") {
    throw new BrowserProtocolError("INVALID_PARAMS", "Result envelope id is required")
  }
  return input.id
}
