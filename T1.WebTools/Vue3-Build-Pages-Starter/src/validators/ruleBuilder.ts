import type { IValidator } from '@/validators/types'

export interface FieldRule {
  validator: IValidator
  error: string
}

export interface IRuleBuilder {
  validate( input: unknown ): string | boolean
}

export class RuleBuilder implements IRuleBuilder {
  constructor( private rules: FieldRule[] ) {
  }

  validate( input: unknown ): string | boolean {
    for (const rule of this.rules) {
      const isValid = rule.validator.isValid(input)
      if (!isValid) {
        return rule.error
      }
    }
    return true
  }
}