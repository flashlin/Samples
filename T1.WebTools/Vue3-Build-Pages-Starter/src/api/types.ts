export interface LoginRequest {
  loginName: string
  password: string
  goToProductSubPage: string
}

export interface LoginResponse {
  Status: LoginStatus
  RedirectUrl: string
  ErrorMessage: string
}

export enum LoginStatus {
  success = 0,
  generalError = 1,
  selfExcluded = 2,
  invalidLoginNameOrPassword = 3,
  invalidBrandType = 4,
  captcha = 5,
  forceChangePassword = 6,
}
