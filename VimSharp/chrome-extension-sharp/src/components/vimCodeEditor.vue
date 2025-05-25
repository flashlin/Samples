<script setup lang="ts">
// https://www.npmjs.com/package/monaco-editor-vue3
import { ref, watch } from 'vue'
import MonacoEditor from 'monaco-editor-vue3'
import * as monaco from 'monaco-editor'
import { initVimMode } from 'monaco-vim'

const props = defineProps<{ value: string }>()
const innerValue = ref(props.value)
const monacoRef = ref<any>(null)

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

<template>
  <div class="vim-editor-root" style="height: 600px;">
    <MonacoEditor
      ref="monacoRef"
      v-model:value="innerValue"
      language="typescript"
      theme="vs-dark"
      height="100%"
      @editorDidMount="onEditorMount"
    />
  </div>
</template>

<style scoped>
.vim-editor-root {
  height: 100%;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  overflow: hidden;
}

.monaco-editor,
.monaco-editor-background,
.monaco-editor .overflow-guard {
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  border: 1px solid #222;
  background: #181c24 !important;
  padding: 8px 0 0 0;
}

/* 狀態列樣式 */
.monaco-vim-status {
  position: absolute;
  right: 16px;
  bottom: 12px;
  background: rgba(30, 30, 30, 0.85);
  color: #ffeb3b;
  font-size: 14px;
  padding: 2px 12px;
  border-radius: 6px;
  z-index: 20;
  pointer-events: none;
  font-family: 'JetBrains Mono', 'Fira Mono', 'Consolas', monospace;
  letter-spacing: 1px;
}
</style> 