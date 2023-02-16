import type { IValidator } from '@/validators/types'
import { isNullOrUndefined, isObject } from '@/utils/shared'

export class RegexValidator implements IValidator {
  constructor( private pattern: RegExp ) {
  }

  isValid( value: unknown ): boolean {
    if (isNullOrUndefined(value) || isObject(value) || Array.isArray(value)) {
      return false
    }
    return this.pattern.test(String(value))
  }
}