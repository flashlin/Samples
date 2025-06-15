<template>
  <div class="vim-editor-root" style="height: 600px;">
    <div ref="editorRoot" class="" style="height:100%"></div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch, nextTick } from 'vue'
import { EditorView, keymap, ViewUpdate } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { lineNumbers, highlightActiveLineGutter } from '@codemirror/view'
import { defaultKeymap } from '@codemirror/commands'
import { autocompletion, CompletionContext, CompletionResult, Completion } from '@codemirror/autocomplete'
import { oneDark } from '@codemirror/theme-one-dark'
import { startCompletion } from '@codemirror/autocomplete'

interface IntellisenseItem {
  title: string
  context: string
}

interface VimCodeEditorProps {
  modelValue: string
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

interface IntellisenseItem {
  title: string
  context: string
}

function codemirrorCompletion(context: CompletionContext): CompletionResult | null {
  if (!suggestionsRef.value.length) return null
  console.log('aaa', suggestionsRef.value)
  return {
    from: context.pos, // 直接用游標位置
    options: suggestionsRef.value.map((item, idx) => ({
      label: item.title,
      apply: item.context
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

function handleCtrlJ(e: KeyboardEvent) {
  if (e.metaKey && (e.key === 'j' || e.key === 'J')) {
    e.preventDefault()
    showIntellisense([
      { title: 'a', context: 'from customer' },
      { title: 'b', context: 'You are winner' }
    ])
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleCtrlJ)
  if (editorRoot.value) {
    view = new EditorView({
      state: EditorState.create({
        doc: innerValue.value,
        extensions: [
          lineNumbers(),
          highlightActiveLineGutter(),
          keymap.of([...defaultKeymap]),
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
  window.removeEventListener('keydown', handleCtrlJ)
  if (view) {
    view.destroy()
    view = null
  }
})
</script>

<style scoped>
.codemirror-editor {
  height: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  border: 1px solid #222;
  background: #181c24 !important;
  padding: 8px 0 0 0;
}
</style>

 