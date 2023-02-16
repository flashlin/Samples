import type { FunctionDirective } from 'vue'

declare module 'vue' {
  export interface ComponentCustomProperties {
    /* built-in API, don't touch! */
    'vGtmEvent:click': FunctionDirective<HTMLElement>
    'vGtmEvent:view': FunctionDirective<HTMLElement>
    'vGtmAttr': FunctionDirective<HTMLElement>
  }
}
