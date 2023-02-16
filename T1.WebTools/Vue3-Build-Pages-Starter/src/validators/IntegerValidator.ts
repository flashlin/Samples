import type { IValidator } from '@/validators/types'

class IsIntegerValidator implements IValidator {
  isValid( value: unknown ): value is number {
    return /^-?[0-9]+$/.test(String(value))
  }
}

export const useIntegerValidator = () => ({
  isInteger: () => new IsIntegerValidator(),
})