import { createApp } from 'vue'
import { I18nFactory } from '@/plugins/i18n'
import { createPinia } from 'pinia'
import { useRegisterStore } from '@/pages/register/stories/registerStore'
import { createDatasetProps } from '@/composable/useDatasetProps'
import { createVeeValidate } from '@/plugins/veeValidate'
import type { RegisterAppProps } from '@/pages/register/types'
import App from "@/pages/register/App.vue"
import { LanguageType } from '@/constants/language'
import { createSimpleGtm } from '@/pages/register/core/simpleGtm'
import defaultResource from '@/pages/register/locales/en.json'

const mountId = '#register-app'
const resources = import.meta.glob('@/pages/register/locales/*.json')

async function initApp() {
  const app = createApp(App)
  app.use(createPinia())
  const entryProps = createDatasetProps<RegisterAppProps>(mountId, {
    btag: '',
    landing: '',
    isRegisterable: true,
    refNo: '',
    brand: BrandType.SBOBET,
    isBtagMatchIaFormat: false,
    loginName: '',
    password: '',
    language: LanguageType.EN,
    infoCenter: '',
    country: '--',
    isFormApp: false,
    clientId: '',
    redirectUri: '',
    promotionCode: '',
    platform: 'm',
  })

  const simpleGtmPlugin = createSimpleGtm({
    enabled: entryProps.state.country.toUpperCase() !== 'IN',
    category: 'Register',
    extendData: {
      event: 'b2c',
      country: entryProps.state.country,
    },
  })
  app.use(entryProps)
  app.use(createVeeValidate())
  app.use(simpleGtmPlugin)
  useRegisterStore().fetchInitialInfo(entryProps.state.language)
  const i18nInstance = await I18nFactory.create(defaultResource, resources, entryProps.state.language)
  app.use(i18nInstance)

  return app
}

initApp().then(( app ) => {
  app.mount(mountId)
})
