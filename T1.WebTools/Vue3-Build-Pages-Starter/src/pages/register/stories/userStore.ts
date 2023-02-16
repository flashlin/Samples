import { defineStore } from 'pinia'
import type { ProductType } from '@/constants/product'
import type { RegistrationLanguageType } from '@/pages/register/types'

interface UserState {
  username: string
  password: string
  landingUrl: string
  isSubmitting: boolean
}

interface GotoAppAsiDto {
  client_id: string
  redirect_uri: string
}

interface GotoDesktopAsiDto {
  product: ProductType
  language: RegistrationLanguageType
}

const userStore = defineStore('USER_STORE', {
  state: (): UserState => ({
    username: '',
    password: '',
    landingUrl: '',
    isSubmitting: false,
  }),
  actions: {
    updateLoginInfo( userLoginState: Partial<UserState> ) {
      this.$patch(userLoginState)
    },
    async gotoAppAsiAsync( appAsiDto: GotoAppAsiDto ) {
      if (this.isSubmitting) return

      try {
        this.isSubmitting = true
        const authClient = new AuthClient({
          landingDomain: this.landingUrl,
        })

        await authClient.processAppLoginAsync({
          username: this.username,
          password: this.password,
          client_id: appAsiDto.client_id,
          redirect_uri: appAsiDto.redirect_uri,
        })

      }
      finally {
        setTimeout(() => {
          this.isSubmitting = false
        }, 500)
      }

    },
    async gotoPlaySiteAsync( requestDto: GotoDesktopAsiDto ) {
      if (this.isSubmitting) return

      try {
        this.isSubmitting = true
        const authClient = new AuthClient({
          landingDomain: this.landingUrl,
        })

        await authClient.processLoginAsync({
          username: this.username,
          password: this.password,
          product: requestDto.product,
          language: requestDto.language,
        })
      }
      catch (e) {
        console.log(e)
        location.href = this.landingUrl
      }
      finally {
        setTimeout(() => {
          this.isSubmitting = false
        }, 500)
      }
    },
  },
})

export const useUserStore = () => userStore()