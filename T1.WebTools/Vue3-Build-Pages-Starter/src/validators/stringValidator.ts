import type { IValidator } from '@/validators/types'
import { isNullOrUndefined } from '@/utils/shared'

interface IStringLengthValidator extends IValidator {
  max( value: number ): this

  min( value: number ): this
}

class StringLengthValidator implements IStringLengthValidator {
  private maxLength?: number
  private minLength?: number

  max( value: number ) {
    this.maxLength = value
    return this
  }

  min( value: number ) {
    this.minLength = value
    return this
  }

  isValid( value: unknown ): boolean {
    if (typeof value === 'string') {
      return this.validateMin(value) && this.validateMax(value)
    }
    return false
  }

  private validateMax( value: string ) {
    if (isNullOrUndefined(this.maxLength)) {
      return true
    }
    return value.length <= this.maxLength
  }

  private validateMin( value: string ) {
    if (isNullOrUndefined(this.minLength)) {
      return true
    }
    return value.length >= this.minLength
  }
}

class StringNotContainWhiteSpaceValidator implements IValidator {
  isValid( value: unknown ): boolean {
    return !/\s/.test(String(value))
  }
}

class StringIsNotEmptyOrWhiteSpaceValidator implements IValidator {
  isValid( value: unknown ): boolean {
    if (isNullOrUndefined(value)) return false

    const inputValue = String(value)
    return new StringNotContainWhiteSpaceValidator().isValid(inputValue) && inputValue.length > 0
  }
}

class StringNotContainNumberValidator implements IValidator{
  isValid( value: unknown ): boolean {
    return !/\d/.test(String(value))
  }
}

export const useStringValidator = () => {
  return {
    length: () => new StringLengthValidator(),
    notContainWhiteSpace: () => new StringNotContainWhiteSpaceValidator(),
    notEmptyOrWhiteSpace: () => new StringIsNotEmptyOrWhiteSpaceValidator(),
    notContainNumber: () => new StringNotContainNumberValidator(),
  }
}