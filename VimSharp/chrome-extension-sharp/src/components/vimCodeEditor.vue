<script setup lang="ts">
// https://www.npmjs.com/package/monaco-editor-vue3
import { ref, watch } from 'vue'
import MonacoEditor from 'monaco-editor-vue3'
import * as monaco from 'monaco-editor'
import { initVimMode2, VimMode2 } from '@/tools/monaco-vim2'

interface VimCodeEditorProps {
  modelValue: string
  enableVim: boolean
}

const props = withDefaults(defineProps<VimCodeEditorProps>(), {
  enableVim: false
})
const emit = defineEmits(['update:modelValue'])
const innerValue = ref(props.modelValue)
const monacoRef = ref<any>(null)

// Sync prop to local value
watch(() => props.modelValue, newValue => {
  if (newValue !== innerValue.value) {
    innerValue.value = newValue
  }
})

// Sync local value to parent
watch(innerValue, (newValue) => {
  emit('update:modelValue', newValue)
})

let vimMode: VimMode2 | null = null
function onEditorMount(editor: monaco.editor.IStandaloneCodeEditor) {
  if (!props.enableVim) return
  if (!vimMode) {
    editor.focus()
    const statusNode = document.createElement('div')
    statusNode.style.cssText = 'position:absolute;right:10px;bottom:10px;color:#fff;z-index:10;'
    editor.getDomNode()?.appendChild(statusNode)
    editor.focus()
    vimMode = initVimMode2(editor, statusNode)
  }
}

/**
 * Get the current editor instance from monacoRef
 */
function getEditorInstance(): monaco.editor.IStandaloneCodeEditor | null {
  // monacoRef.value?.editor for monaco-editor-vue3
  return monacoRef.value?.editor || null
}

/**
 * Get the content of the current line with cursor position
 * @returns [leftContent, rightContent]
 */
function getCurrentLineWithCursor(): string[] {
  const editor = getEditorInstance()
  if (!editor) return ['', '']
  const position = editor.getPosition()
  if (!position) return ['', '']
  const model = editor.getModel()
  if (!model) return ['', '']
  const lineContent = model.getLineContent(position.lineNumber)
  const cursorIndex = position.column - 1 // column is 1-based
  const left = lineContent.slice(0, cursorIndex)
  const right = lineContent.slice(cursorIndex)
  return [left, right]
}

defineExpose({
  getCurrentLineWithCursor
})
</script>

<template>
  <div class="vim-editor-root w-full h-full">
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