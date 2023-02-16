import { RuleBuilder } from '@/validators/ruleBuilder'
import { useStringValidator } from '@/validators/stringValidator'
import { RegexValidator } from '@/validators/regexValidator'

export function createMobileRules(): RuleBuilder {
  return new RuleBuilder([{
    validator: useStringValidator().length().min(5),
    error: 'MobileLength',
  },
  {
    validator: new RegexValidator(/^\d[\d]{3,}\d$/),
    error: 'MobileNumeric',
  },
  ])
}
