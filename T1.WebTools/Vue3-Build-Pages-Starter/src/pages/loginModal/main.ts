import { createApp } from 'vue'
import type { LoginModalProps } from '@/pages/loginModal/types'
import App from "@/pages/loginModal/App.vue"
import { LanguageType } from '@/constants/language'
import { createDatasetProps } from '@/composable/useDatasetProps'
import { I18nFactory } from '@/plugins/i18n'
const resources = import.meta.glob('@/pages/loginModal/locales/*.json')

const mountId = '#login-modal-app'

async function initApp() {
  const entryProps = createDatasetProps<LoginModalProps>(mountId, {
    language: LanguageType.EN,
    registerUrl: "",
    accountAssistanceUrl: "",
    contactUsUrl: "",
    loginTarget: "",
  })
  const app = createApp(App)
  const i18nInstance = await I18nFactory.createDefault(resources, entryProps.state.language)
  app.use(entryProps)
  app.use(i18nInstance)

  return app
}

initApp().then(( app ) => {
  app.mount(mountId)
})
