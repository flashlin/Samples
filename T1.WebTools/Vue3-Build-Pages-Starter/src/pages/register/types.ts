import { BrandType } from '@/constants/brandType'
import { LanguageType } from '@/constants/language'

export type RegistrationLanguageType = Exclude<LanguageType, LanguageType.HI_IN | LanguageType.TA_IN | LanguageType.PT_BR>

export interface RegisterAppProps {
  refNo: string
  language: RegistrationLanguageType
  loginName: string
  password: string
  brand: BrandType
  isRegisterable: boolean
  landing: string
  infoCenter: string
  isBtagMatchIaFormat: boolean
  btag: string
  country: string
  isFormApp: boolean
  clientId: string
  redirectUri: string
  promotionCode: string
  platform: string
}

export interface AccountInfo {
  loginName: string
  password: string
  refNo: string
  btag: string
  currency?: string
}

export interface PersonalInfo {
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
}

export const enum CaptchaType {
  SixNumeric = 'SixNumeric',
  NumericAddition = 'NumericAddition',
}

export interface RegisterSuccessDetail {
  hashedCustId: string
  isoCurrency: string
}
