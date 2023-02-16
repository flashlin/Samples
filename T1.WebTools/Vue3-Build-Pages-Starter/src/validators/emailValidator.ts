import type { IValidator } from '@/validators/types'

class EmailFormatValidator implements IValidator {
  private EMAIL_FORMAT_REGEX = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
  isValid( value: unknown ): boolean {
    return this.EMAIL_FORMAT_REGEX.test(String(value))
  }
}

export const useEmailValidator = () => ({
  isValidFormat: () => new EmailFormatValidator(),
})