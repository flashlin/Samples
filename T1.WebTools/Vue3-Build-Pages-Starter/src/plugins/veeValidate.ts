import type { App } from 'vue'
import { ErrorMessage, Field, Form } from 'vee-validate'

export const createVeeValidate = () => {
  return {
    install: ( app: App ) => {
      // eslint-disable-next-line vue/no-reserved-component-names
      app.component('Form', Form)
      app.component('Field', Field)
      app.component('ErrorMessage', ErrorMessage)
    },
  }
}