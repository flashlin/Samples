<script lang="ts" setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
 
import { createUniver, defaultTheme, FUniver, LocaleType, merge, Univer } from '@univerjs/presets';
import { UniverSheetsCorePreset } from '@univerjs/presets/preset-sheets-core';
import UniverPresetSheetsCoreZhTW from '@univerjs/presets/preset-sheets-core/locales/zh-TW';
 
import '@univerjs/presets/lib/styles/preset-sheets-core.css';
 
const container = ref<HTMLElement | null>(null)
 
let univerInstance: Univer | null = null
let univerAPIInstance: FUniver | null = null
 
onMounted(() => {
  const { univer, univerAPI } = createUniver({
    locale: LocaleType.ZH_TW,
    locales: {
      [LocaleType.ZH_TW]: merge(
        {}, 
        UniverPresetSheetsCoreZhTW
      ),
    },
    theme: defaultTheme,
    presets: [
      UniverSheetsCorePreset({
        container: container.value as HTMLElement,
      }),
    ],
  })
 
  univerAPI.createWorkbook({ name: 'Test Sheet' })
 
  univerInstance = univer
  univerAPIInstance = univerAPI
})
 
onBeforeUnmount(() => {
  univerInstance?.dispose()
  univerAPIInstance?.dispose()
  univerInstance = null
  univerAPIInstance = null
})
</script>

<template>
   <div ref="container" class="excel-container"></div>
</template>

<style scoped>
.excel-container {
  width: 100%;
  height: 100%;
}
</style>
