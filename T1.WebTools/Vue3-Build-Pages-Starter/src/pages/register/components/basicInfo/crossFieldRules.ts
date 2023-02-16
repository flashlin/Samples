import type { IValidator } from '@/validators/types'
import type { Ref } from 'vue'
import type { FieldRule } from '@/validators/ruleBuilder'

class NotContainRefStringValidator implements IValidator {
  constructor( private stringRef: Ref<string> ) {
  }

  isValid( value: unknown ): boolean | Promise<boolean> {
    if (!this.stringRef.value || this.stringRef.value.length <= 0) return true

    return !String(value).includes(this.stringRef.value)
  }
}

export const createPasswordExtendRules = ( loginNameRef: Ref<string> ): FieldRule[] => {
  return [{
    validator: new NotContainRefStringValidator(loginNameRef),
    error: 'PasswordShouldNotContainLoginName',
  }]
}
