import { RuleBuilder } from '@/validators/ruleBuilder'
import { useAlphabetValidator } from '@/validators/alphaValidator'
import { useStringValidator } from '@/validators/stringValidator'
import type { IValidator } from '@/validators/types'

class LoginNameValidator implements IValidator {
  isValid( value: unknown ): boolean {
    if (typeof value === 'string') {
      return this.containAlphabetNumber(value)
    }
    return false
  }

  private containAlphabetNumber( value: string ) {
    return /^(?=.*[a-z])(?=.*[0-9])[a-z0-9]{5,14}$/i.test(value)
  }
}

class UnavailableLoginNameValidator implements IValidator {
  private validatedFailedLoginName: string[] =[]

  isValid( value: unknown ): boolean {
    if(typeof value === 'string'){
      return !this.validatedFailedLoginName.includes(value)
    }
    return false
  }
  addValidatedFailedLoginName(loginName: string){
    this.validatedFailedLoginName.push(loginName)
  }
}

export function useLoginNameRules() {
  const unavailableLoginNameValidator = new UnavailableLoginNameValidator()
  const ruleBuilder = new RuleBuilder([
    {
      validator: unavailableLoginNameValidator,
      error: 'LoginNameNotAvailable',
    },
    {
      validator: useAlphabetValidator().isStartsWithAlphabet(),
      error: 'LoginNameStartWithAlphabet',
    },
    {
      validator: useStringValidator().length().min(5).max(14),
      error: 'LoginNameLength',
    },
    {
      validator: new LoginNameValidator(),
      error: 'LoginNameContainAlphabetNumber',
    },
  ])
  return {
    validate: (input: unknown) => ruleBuilder.validate(input),
    addValidatedFailedLoginName: (loginName: string) => unavailableLoginNameValidator.addValidatedFailedLoginName(loginName),
  }
}