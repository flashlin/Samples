<script lang="ts" setup>
// https://docs.univer.ai/zh-CN/guides/sheets/getting-started/installation
// https://docs.univer.ai/zh-CN/guides/sheets/integrations/vue
import { onMounted, onBeforeUnmount, ref } from 'vue'
 
import '@univerjs/presets/lib/styles/preset-sheets-core.css';
 
const container = ref<HTMLElement | null>(null)
 
let univerInstance: any = null
let univerAPIInstance: any = null
 
onMounted(async () => {
  // 動態 import 各模組
  const presets = await import('@univerjs/presets');
  const { createUniver, defaultTheme, LocaleType, merge } = presets;
  const { UniverSheetsCorePreset } = await import('@univerjs/presets/preset-sheets-core');
  const UniverPresetSheetsCoreZhTW = (await import('@univerjs/presets/preset-sheets-core/locales/zh-TW')).default;

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
