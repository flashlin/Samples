<script lang="ts" setup>
import { ref } from 'vue'
import { useLoginNameRules } from '@/components/fields/loginNameRules'
import { useEventBus } from '@vueuse/core'
import { SetLoginNameUnavailableEvent } from '@/components/fields/fieldEvents'

interface LoginNameProps {
  defaultLoginName?: string
}

const props = defineProps<LoginNameProps>()
const eventBus = useEventBus(SetLoginNameUnavailableEvent)
//const loginNameInputRef = ref<InstanceType<typeof AppFormField>>()
const loginNameRef = ref<string>(props.defaultLoginName || '')
const suggestions = ref<string[]>([])
const isShowLoginNameTips = ref<boolean>(true)
const loginNameRules = useLoginNameRules()

const onSelectLoginName = ( value: string) => {
  loginNameRef.value = value
  suggestions.value = []
}

const setCurrentLoginNameUnavailable = ( suggestionLoginNames: string[] ) => {
  suggestions.value = suggestionLoginNames
  loginNameRules.addValidatedFailedLoginName(loginNameRef.value)
  //loginNameInputRef.value?.setFocus()
  //loginNameInputRef.value?.validate()
}

eventBus.on(({ suggestLists }) => {
  setCurrentLoginNameUnavailable(suggestLists)
})
</script>
<template>
</template>
