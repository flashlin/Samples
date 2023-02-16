import { type IRequestProxy, createHttpClient } from "@/plugins/request"
import type { LoginRequest, LoginResponse } from "./types"
import { LoginStatus } from "./types"

export interface ILoginApiClient {
  loginAsync(request: LoginRequest): Promise<LoginResponse>

}

class LoginApiClient implements ILoginApiClient {
  constructor( private readonly httpClient: IRequestProxy ) {
  }
  loginAsync(request: LoginRequest): Promise<LoginResponse> {
    return this.httpClient.post({
      url: "/promo/login",
      data: request,
    })
  }
}

class MockLoginApiClient implements ILoginApiClient {
  loginAsync(_request: LoginRequest): Promise<LoginResponse> {
    return Promise.resolve({
      Status: LoginStatus.success,
      RedirectUrl: "#",
      ErrorMessage: "",
    })
  }
  
}

const useLoginApiClient = (): ILoginApiClient => {
  if (import.meta.env.MODE === 'development') {
    return new MockLoginApiClient()
  }
  return new LoginApiClient(createHttpClient())
}

export default useLoginApiClient()