<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { EditorView, keymap, ViewUpdate } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { lineNumbers, highlightActiveLineGutter } from '@codemirror/view'
import { defaultKeymap } from '@codemirror/commands'
import { autocompletion, CompletionContext, CompletionResult, closeCompletion } from '@codemirror/autocomplete'
import { oneDark } from '@codemirror/theme-one-dark'
import { startCompletion } from '@codemirror/autocomplete'
import type { IntellisenseItem, OnShowIntellisenseFn } from './CodeEditorTypes'

interface VimCodeEditorProps {
  modelValue: string
  onShowIntellisense?: OnShowIntellisenseFn
}

const props = withDefaults(defineProps<VimCodeEditorProps>(), {})
const emit = defineEmits(['update:modelValue'])
const innerValue = ref(props.modelValue)
const suggestionsRef = ref<IntellisenseItem[]>([])
// 新增 loading 狀態
const isLoading = ref(false)

// 雙向綁定
watch(() => props.modelValue, newValue => {
  if (newValue !== innerValue.value) {
    innerValue.value = newValue
    if (view) {
      const cur = view.state.selection.main;
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: newValue },
        selection: { anchor: Math.min(cur.anchor, newValue.length) }
      })
    }
  }
})
watch(innerValue, (newValue) => {
  emit('update:modelValue', newValue)
})

const editorRoot = ref<HTMLElement | null>(null)
let view: EditorView | null = null

// 簡單模糊比對函式
function fuzzyMatch(text: string, pattern: string) {
  let t = 0, p = 0
  text = text.toLowerCase()
  pattern = pattern.toLowerCase()
  while (t < text.length && p < pattern.length) {
    if (text[t] === pattern[p]) p++
    t++
  }
  return p === pattern.length
}

function codemirrorCompletion(context: CompletionContext): CompletionResult | null {
  if (isLoading.value) {
    return {
      from: context.pos,
      options: [{
        label: 'Loading...',
        apply: () => false // 不可選
      }]
    }
  }
  if (!suggestionsRef.value.length) return null

  // 取得目前游標前的文字
  const word = context.matchBefore(/[^\s]*/)?.text ?? ''
  // 多字詞切割
  const keywords = word.trim().split(/\s+/).filter(Boolean)
  // 多字詞模糊比對
  const filtered = suggestionsRef.value.filter(item =>
    keywords.every(kw => fuzzyMatch(item.title, kw))
  )

  // 沒有符合就全部顯示
  const suggestionList = filtered.length ? filtered : suggestionsRef.value

  return {
    from: context.pos - word.length,
    options: suggestionList.map((item, _idx) => ({
      label: item.title,
      apply: (view: EditorView, _completion: any, from: number, to: number) => {
        const realFrom = item.getFromPosition ? item.getFromPosition(from) : from
        const realTo = to
        const insertText = item.getContext()
        view.dispatch({
          changes: { from: realFrom, to: realTo, insert: insertText },
          selection: { anchor: realFrom + insertText.length }
        })
        closeCompletion(view)
        return true
      }
    }))
  }
}

// 顯示 loading intellisense
function showLoadingIntellisense() {
  isLoading.value = true
  suggestionsRef.value = [{
    title: 'Loading...',
    getContext: () => '',
    getFromPosition: (from: number) => from
  }]
  if (view) {
    startCompletion(view)
  }
}

function showIntellisense(items: IntellisenseItem[]): void {
  suggestionsRef.value = items
  if (view) {
    view.dispatch({
      effects: [],
      scrollIntoView: true
    })
    startCompletion(view)
  }
}

async function handleShowIntellisense() {
  if (isLoading.value) return // 正在 loading 就不再觸發
  showLoadingIntellisense()
  // 取得游標前後 context
  const [before, after] = getContextWithCursor()
  let items: IntellisenseItem[] = [{
    title: '<No Suggestions>',
    getContext: () => '',
  }]
  if (props.onShowIntellisense) {
    // 呼叫外部 delegate
    try{ 
      const result = await props.onShowIntellisense({ content: [before, after] })
      if (Array.isArray(result)) items = result
    } catch (error) {
      console.error('handleShowIntellisense error', error)
    }
  }
  isLoading.value = false
  showIntellisense(items)
}

const customKeymap = [
  {
    key: "Mod-j", // Mod 代表 Ctrl(Win/Linux) 或 Cmd(Mac)
    run: () => {
      handleShowIntellisense()
      return true // 阻止預設行為
    }
  },
  {
    key: "Mod-k",
    run: () => {
      if (view) closeCompletion(view)
      return true
    }
  }
]

function getContextWithCursor(): [string, string] {
  if (!view) return ['', '']
  const state = view.state
  const pos = state.selection.main.head
  const doc = state.doc.toString()
  return [doc.slice(0, pos), doc.slice(pos)]
}

onMounted(() => {
  if (editorRoot.value) {
    view = new EditorView({
      state: EditorState.create({
        doc: innerValue.value,
        extensions: [
          lineNumbers(),
          highlightActiveLineGutter(),
          keymap.of([...defaultKeymap, ...customKeymap]),
          oneDark,
          EditorView.lineWrapping,
          EditorView.updateListener.of((v: ViewUpdate) => {
            if (v.docChanged) {
              innerValue.value = v.state.doc.toString()
            }
          }),
          autocompletion({
            override: [codemirrorCompletion],
            activateOnTyping: false // 僅手動觸發
          })
        ]
      }),
      parent: editorRoot.value
    })
  }
})

onUnmounted(() => {
  if (view) {
    view.destroy()
    view = null
  }
})

// expose 方法
defineExpose({
  getContextWithCursor
})
</script>

<template>
  <div class="vim-editor-root" style="height: 600px;">
    <div ref="editorRoot" class="" style="height:100%"></div>
  </div>
</template>

<style scoped>
.codemirror-editor {
  height: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  border: 1px solid #222;
  background: #181c24 !important;
  padding: 8px 0 0 0;
}

/* 覆蓋 CodeMirror completion 的暗黑主題樣式 */
:deep(.cm-tooltip-autocomplete) {
  background: #181c24 !important;
  color: #c9d1d9 !important;
  border: 1px solid #222 !important;
}
:deep(.cm-tooltip-autocomplete .cm-completionLabel) {
  color: #c9d1d9 !important;
}
:deep(.cm-tooltip-autocomplete .cm-completionIcon) {
  color: #8b949e !important;
}
:deep(.cm-tooltip-autocomplete .cm-completionSelected) {
  background: #388bfd !important;
  color: #fff !important;
}
</style>

 