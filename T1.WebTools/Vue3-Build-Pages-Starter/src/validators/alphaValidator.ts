import type { IValidator } from '@/validators/types'

class StartWithAlphabetValidator implements IValidator {
  isValid( value: unknown ): boolean {
    if (typeof value == 'string') {
      return /^[a-zA-Z]+.*$/i.test(value)
    }
    return false
  }
}

export const useAlphabetValidator = () => {
  return{
    isStartsWithAlphabet: () => new StartWithAlphabetValidator(),
  }
}