<script setup lang="ts">
// https://www.npmjs.com/package/monaco-editor-vue3
import { ref, watch, onMounted, onUnmounted } from 'vue'
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

/**
 * 取得從第一個字元到 cursor 前一個字元，以及 cursor 後到內容結尾的字串
 * @returns [leftContent, rightContent]
 */
function getContextWithCursor(): string[] {
  const editor = getEditorInstance()
  if (!editor) return ['', '']
  const position = editor.getPosition()
  if (!position) return ['', '']
  const model = editor.getModel()
  if (!model) return ['', '']
  const fullText = model.getValue()
  // 計算 cursor 在全文的 offset
  const offset = model.getOffsetAt(position)
  const left = fullText.slice(0, offset)
  const right = fullText.slice(offset)
  return [left, right]
}

// IntellisenseItem 型別
interface IntellisenseItem {
  title: string
  context: string
}

// suggestionsRef 用於動態提供 suggestions
const suggestionsRef = ref<any[]>([])
let providerDispose: monaco.IDisposable | null = null

onMounted(() => {
  providerDispose = monaco.languages.registerCompletionItemProvider('typescript', {
    provideCompletionItems: (model, position) => {
      // 產生 range
      const word = model.getWordAtPosition(position)
      const range = word
        ? new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn)
        : new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column)
      // 將 suggestionsRef 內容包裝成 CompletionItem
      const suggestions = suggestionsRef.value.map((item, idx) => ({
        label: item.title,
        kind: monaco.languages.CompletionItemKind.Text,
        insertText: item.context,
        sortText: idx.toString().padStart(4, '0'),
        range
      }))
      return { suggestions }
    },
    triggerCharacters: ['.']
  })
})

onUnmounted(() => {
  if (providerDispose) {
    providerDispose.dispose()
    providerDispose = null
  }
})

/**
 * 於游標位置顯示智慧感知清單，選取後插入 context
 */
function showIntellisense(items: IntellisenseItem[]): void {
  const editor = getEditorInstance()
  if (!editor) return
  suggestionsRef.value = items
  editor.trigger('keyboard', 'editor.action.triggerSuggest', {})
}

defineExpose({
  getCurrentLineWithCursor,
  getContextWithCursor,
  showIntellisense
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