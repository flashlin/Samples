import {
  handleAuthenticatedJsonRequest,
  type HttpRouteSecurity
} from "../http-response"

export function createHealthRoute(security: HttpRouteSecurity): (request: Request) => Response {
  return request => handleAuthenticatedJsonRequest(request, security, () => ({
    status: "ok"
  }))
}
