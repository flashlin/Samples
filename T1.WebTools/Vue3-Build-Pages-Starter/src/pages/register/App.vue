<script lang="ts" setup>
import { useEntryProps } from '@/composable/useDatasetProps'
import type { RegisterAppProps } from '@/pages/register/types'
import { onMounted } from 'vue'
import { useSimpleGtm } from '@/pages/register/core/simpleGtm'

const { refNo } = useEntryProps<RegisterAppProps>()
const simpleGtm = useSimpleGtm()
onMounted(() => {
  simpleGtm.pushEvent('start', {
    label: 'registration_start',
  })
  simpleGtm.pushEvent('view', {
    label: 'pageview_newonepage',
  })
})

//const isTestLoginFormEnabled = import.meta.env.DEV && location.search.includes('test_login_btn')

</script>

<template>
  <h1 class="main-title"> {{ $t("Registration") }}</h1>
  <div class="main-subText">
    {{ $t("ReferenceId") }} <span id="RefNo">{{ refNo }}</span>
  </div>
</template>
