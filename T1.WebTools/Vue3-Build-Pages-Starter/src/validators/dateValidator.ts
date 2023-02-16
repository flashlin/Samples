import type { IValidator } from '@/validators/types'
import { useIntegerValidator } from '@/validators/IntegerValidator'

class DayValidator implements IValidator {
  isValid( day: unknown ): boolean {
    return useIntegerValidator().isInteger().isValid(day) && day >= 1 && day <= 31
  }
}

class MonthValidator implements IValidator {
  isValid( month: unknown ): boolean {
    return useIntegerValidator().isInteger().isValid(month) && month >= 1 && month <= 12
  }
}

export const useDateValidator = () => ({
  day: () => new DayValidator(),
  month: () => new MonthValidator(),
})