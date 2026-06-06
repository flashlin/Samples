import type {
  BrowserCommandParams,
  BrowserCommandResults,
  BrowserMethod
} from "@page-probe/protocol"
import type { CommandDispatcher } from "../command-bus/command-dispatcher"

export class BrowserToolService {
  constructor(private readonly dispatcher: CommandDispatcher) {}

  execute<TMethod extends BrowserMethod>(
    method: TMethod,
    params: BrowserCommandParams[TMethod],
    clientId?: string
  ): Promise<BrowserCommandResults[TMethod]> {
    return this.dispatcher.dispatch(method, params, clientId)
  }
}
