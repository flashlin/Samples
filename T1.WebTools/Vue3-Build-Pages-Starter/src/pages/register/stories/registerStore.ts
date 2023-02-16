import { defineStore } from 'pinia'
import { fetchRegisterInitialInfoAsync } from '@/pages/register/api/registerApiClient'
import type { CountryInfo, SecurityQuestion } from '@/pages/register/api/types'
import { useEventHelper } from '@/composable/useEventHelper'

interface RegisterState {
  securityQuestions: SecurityQuestion[]
  allowCountries: CountryInfo[]
  selectedCountryInfo: CountryInfo
  token: string
  isShowCaptcha: boolean
}

const registerStore = defineStore('REGISTER_STORE', {
  state: (): RegisterState => ({
    securityQuestions: [],
    allowCountries: [],
    token: '',
    selectedCountryInfo: {
      currencies: [],
      countryCode: '--',
      phoneCode: 0,
      countryName: '--',
      license: 'IOM',
    },
    isShowCaptcha: false,
  }),
  getters: {
    isMnl: ( state ) => state.selectedCountryInfo.license.toUpperCase() === "MNL",
    legalAge(): number {
      return this.isMnl ? 21 : 18
    },
    isEmptyCountries: ( state ) => state.allowCountries.length === 0,
  },
  actions: {
    fetchInitialInfo( language: string ) {
      fetchRegisterInitialInfoAsync({ language })
        .then(( { securityQuestions, allowCountries, selectedCountry, token, isShowCaptcha } ) => {
          const countryInfo = allowCountries[selectedCountry] ?? allowCountries[0]
          this.updateCurrentSelectedCountry(countryInfo)
          this.$patch({ securityQuestions, allowCountries, token, isShowCaptcha })
        })
    },
    async refreshTokenAsync() {
      const response = await fetchRegisterInitialInfoAsync({ language: 'en' })
      this.$patch({ token: response.token })
    },
    updateCurrentSelectedCountry( countryInfo: CountryInfo ) {
      this.$patch({ selectedCountryInfo: countryInfo })
      useEventHelper("register").emit('country-changed', { isMnl: this.isMnl })
    },
  },
})

export const useRegisterStore = () => registerStore()