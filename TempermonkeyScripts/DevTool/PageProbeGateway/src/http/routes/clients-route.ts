import type { ConnectionRegistry } from "../../connections/connection-registry"
import {
  handleAuthenticatedJsonRequest,
  type HttpRouteSecurity
} from "../http-response"

export type ClientsRouteDependencies = HttpRouteSecurity & {
  connectionRegistry: Pick<ConnectionRegistry, "list">
}

export function createClientsRoute(
  dependencies: ClientsRouteDependencies
): (request: Request) => Response {
  return request => handleAuthenticatedJsonRequest(request, dependencies, () => ({
    clients: dependencies.connectionRegistry.list()
  }))
}
