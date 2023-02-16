<script lang="ts" setup>
import BaseModal from "@/components/modals/BaseModal.vue"
import { useEntryProps } from '@/composable/useDatasetProps'
import type { RegisterAppProps } from '@/pages/register/types'
import { useUserStore } from '@/pages/register/stories/userStore'
import { type ProductType } from '@/constants/product'
import { useCountdownTimer } from '@/pages/register/composable/useCountdownTimer'
import { storeToRefs } from 'pinia'

const { brand, language } = useEntryProps<RegisterAppProps>()
const store = useUserStore()
const { isSubmitting } = storeToRefs(store)
const brandName = brand.toUpperCase()
const { seconds, cancel: cancelLoginTimer } = useCountdownTimer(() =>
  store.gotoPlaySiteAsync({
    language,
    product: 'landing',
  }), {
  seconds: 10,
})
const gotoAsi = ( product: ProductType ) => {
  cancelLoginTimer()

  store.gotoPlaySiteAsync({
    language,
    product,
  })
}


</script>

<template>
  <BaseModal :title="$t('RegisterSuccess')" modal-type="success" text-class="my-3">
    <template #text>
      <span>{{ $t('Redirect', { seconds, brand: brandName }) }}</span>
    </template>
    <template #footer>
      <button class="t-button t-button-primary t-button-medium"
              data-testid="gotoDepositButton"
              :disabled="isSubmitting"
              v-gtm-event:click="'Register_btn_PopUpDepositNow'"
              @click.prevent="gotoAsi('payment')"
      >{{ $t('DepositNow') }}
      </button>
      <a class="link link-sub text-small mt-4"
         data-testid="gotoLandingButton"
         @click.prevent="gotoAsi('landing')"
         v-gtm-event:click="'Register_btn_PopUpExplorePage'"
      >
        {{ $t('GoToWebsiteNow', { brand: brandName }) }}
      </a>
    </template>
  </BaseModal>
</template>
