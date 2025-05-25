<template>
  <MonacoEditor
    v-model:value="innerValue"
    language="typescript"
    theme="vs-dark"
    height="400px"
    @mount="onEditorMount"
  />
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import MonacoEditor from 'monaco-editor-vue3'
import * as monaco from 'monaco-editor'
import { initVimMode } from 'monaco-vim'

const props = defineProps<{ value: string }>()

const innerValue = ref(props.value)

watch(() => props.value, v => {
  if (v !== innerValue.value) innerValue.value = v
})

let vimMode: any = null
function onEditorMount(editor: monaco.editor.IStandaloneCodeEditor) {
  // 初始化 Vim 模式
  if (!vimMode) {
    const statusNode = document.createElement('div')
    statusNode.style.cssText = 'position:absolute;right:10px;bottom:10px;color:#fff;z-index:10;'
    editor.getDomNode()?.appendChild(statusNode)
    vimMode = initVimMode(editor, statusNode)
  }
}
</script>

<style scoped>
/* 可自訂編輯器外觀 */
</style> 