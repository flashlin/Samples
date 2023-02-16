<script lang="ts" setup>
import type { LoginModalProps } from "@/pages/loginModal/types"
import { ref } from "vue"
import { useEntryProps } from "@/composable/useDatasetProps"
import { useEventHelper } from '@/composable/useEventHelper'
import { isOnApp, postAppMessage } from "@/utils/shared"

const isShow = ref(false)
const {
  registerUrl,
  accountAssistanceUrl,
  contactUsUrl,
  loginTarget,
} = useEntryProps<LoginModalProps>()
const eventHelper = useEventHelper("promotion")

eventHelper.on('open-login-modal', () => {
  openModal()
})

function openModal() {
  if(isOnApp()){
    postAppMessage("world_cup_event_login")
    return
  }
  isShow.value = true
}
</script>

<template>
  <button class="login-btn" data-gtm-label="Dday_Click_Btn_Login" data-gtm-category="Dday" @click="openModal">{{ $t("Login") }}</button>
  <LoginModal class="login-modal"
    v-model:isShow="isShow"
    :register-url="registerUrl"
    :account-assistance-url="accountAssistanceUrl"
    :contact-us-url="contactUsUrl"
    :login-target="loginTarget"
  />
</template>

<style lang="scss"></style>
