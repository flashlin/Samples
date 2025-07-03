<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { EditorView, keymap, ViewUpdate } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { lineNumbers, highlightActiveLineGutter } from '@codemirror/view'
import { defaultKeymap } from '@codemirror/commands'
import { autocompletion, CompletionContext, CompletionResult, closeCompletion } from '@codemirror/autocomplete'
import { oneDark } from '@codemirror/theme-one-dark'
import { startCompletion } from '@codemirror/autocomplete'
import type { IntellisenseItem, OnShowIntellisenseFn } from './codeEditorTypes'

interface VimCodeEditorProps {
  modelValue: string
  onShowIntellisense?: OnShowIntellisenseFn
}

const props = withDefaults(defineProps<VimCodeEditorProps>(), {})
const emit = defineEmits(['update:modelValue'])
const innerValue = ref(props.modelValue)
const suggestionsRef = ref<IntellisenseItem[]>([])

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

function codemirrorCompletion(context: CompletionContext): CompletionResult | null {
  if (!suggestionsRef.value.length) return null
  return {
    from: context.pos, // 直接用游標位置
    options: suggestionsRef.value.map((item, _idx) => ({
      label: item.title,
      apply: item.context === '' ? (view) => {
        closeCompletion(view)
        return true;
       } 
      : item.context
    }))
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
  // 取得游標前後 context
  const [before, after] = getContextWithCursor()
  let items: IntellisenseItem[] = []
  if (props.onShowIntellisense) {
    // 呼叫外部 delegate
    const result = await props.onShowIntellisense({ content: [before, after] })
    if (Array.isArray(result)) items = result
  }  
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

<style>
.codemirror-editor {
  height: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  border: 1px solid #222;
  background: #181c24 !important;
  padding: 8px 0 0 0;
}

/* 覆蓋 CodeMirror completion 的暗黑主題樣式 */
.cm-tooltip-autocomplete {
  background: #181c24 !important; /* 彈出清單背景，與編輯器一致 */
  color: #c9d1d9 !important;      /* 項目文字顏色 */
  border: 1px solid #222 !important;
}
.cm-tooltip-autocomplete .cm-completionLabel {
  color: #c9d1d9 !important;      /* 項目文字顏色 */
}
.cm-tooltip-autocomplete .cm-completionIcon {
  color: #8b949e !important;      /* icon 顏色 */
}
.cm-tooltip-autocomplete .cm-completionSelected {
  background: #388bfd !important; /* 亮藍色 */
  color: #fff !important;         /* 亮色文字 */
}
</style>

 