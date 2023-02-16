import type { CaptchaType } from '@/pages/register/types'
import type { LanguageType } from '@/constants/language'
import type { ProductType } from '@/constants/product'

export interface RegisterRequest {
  token: string
  refNo: string
  loginName: string
  password: string
  isoCurrency: string
  firstName: string
  lastName: string
  day: number
  month: number
  year: number
  email: string
  mobile: string
  contactPreference: string
  nationalityCode: string
  residentCountryCode: string
  promotionCode: string
  promotionEmail: boolean
  gender: string
  securityQuestion: number
  securityAnswer: string
  language: LanguageType
  btag: string
  validationCode?: string
  platform: string
}

export enum RegisterErrorCode {
  LoginNameNotAvailable = 2,
  PasswordInvalid = 3,
  EmailInvalid = 5,
  PromotionCodeNotExist = 22,
  PromotionCodeExpired = 23,
  PromotionCodeNotSupportedCountry = 24,
  PromotionCodeNotSupportedCurrency = 25,
  IncorrectValidationCode = 26,
  InvalidPromotionCode = 39,
}

export interface RegisterSuccessfullyResponse {
  hashedCustId: string
  success: boolean
  isoCurrency: string
  landingUrl: string
  isOauth: boolean
}

export interface RegisterErrorResponse {
  success: boolean
  errorCode: string
  suggestions: string[]
  isShowCaptcha: boolean
}

export type RegisterResponse = RegisterSuccessfullyResponse | RegisterErrorResponse

export interface CountryInfo {
  countryCode: string
  countryName: string
  phoneCode: number
  license: string
  currencies: string[]
}

export interface SecurityQuestion {
  id: number
  question: string
}

export interface RegistrationConfigurationResponse {
  securityQuestions: SecurityQuestion[]
  allowCountries: CountryInfo[]
  selectedCountry: number
  token: string
  isShowCaptcha: boolean
}

export interface RegistrationConfigurationRequest {
  language?: string
}

export interface ValidateNameRequest {
  name: string
  residentCountryCode: string
}

export interface ValidateNameResponse {
  isValid: boolean
}

// export interface GetDepositBonusPromotionLinkRequest {
//   residentCountryCode: string
//   lang: LanguageType
//   promoCode?: string
// }

// export interface CheckLoginNameAvailabilityRequest {
//   loginName: string
//   validationCode?: string
// }
//
// export interface CheckLoginNameAvailabilityResponse {
//   suggestion: string[]
//   isAvailable: boolean
//   showCaptcha: boolean
// }

export interface GetValidationCodeResponse {
  image: string
  type: CaptchaType
}

export interface GotoPlayerSiteResponse {
  targetUrl: string
}

export interface GotoPlayerSiteRequest {
  product: ProductType
  token: string
  language: LanguageType
}

export interface GetTokenResponse {
  access_token: string
  expires_in: number
  token_type: string
  refresh_token: string
  scope: string
}