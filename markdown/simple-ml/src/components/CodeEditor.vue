<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { createEditor, KeyCode, KeyMod } from '@/plugin/MonacoEditor'
import defaultTemplate from '@/assets/templates/default.plantuml?raw'

const CLASS_NAME = 'UmlCodeEditor'

onMounted(async () => {
  const editorElm = document.querySelector<HTMLElement>(`.${CLASS_NAME}`)

  if (!editorElm) {
    return
  }

  const editor = createEditor(editorElm, {
    value: defaultTemplate
  })
  editor.onDidChangeModelContent(async (event: any) => {
    //!event.isFlush && (await update(editor.getValue()))
  })
  editor.addCommand(KeyMod.CtrlCmd | KeyCode.Enter, async () => {
    //await renderHtml(editor.getValue())
  })
  editor.addCommand(KeyMod.WinCtrl | KeyCode.Enter, async () => {
    //await renderHtml(editor.getValue())
  })
});
</script>

<template>
  <div :class="[CLASS_NAME]" class="overflow-hidden" />
</template>
