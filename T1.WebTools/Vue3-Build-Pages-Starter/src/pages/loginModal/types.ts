import { LanguageType } from './../../constants/language'

export interface LoginModalProps {
  language: LanguageType
  registerUrl: string
  accountAssistanceUrl: string
  contactUsUrl: string
  loginTarget: string
}