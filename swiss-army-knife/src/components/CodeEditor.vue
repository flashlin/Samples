<script setup lang="ts">
import { reactive, shallowRef } from 'vue';
import { Codemirror } from 'vue-codemirror'
import { sql } from '@codemirror/lang-sql'
import { oneDark } from '@codemirror/theme-one-dark'

interface ICodeEditorProps {
  modelValue: string
}
const props = withDefaults(defineProps<ICodeEditorProps>(), {
  modelValue: ""
})

interface ICodeEditorEmits {
  (e: 'update:modelValue', value: string): void
}
const emits = defineEmits<ICodeEditorEmits>();

const codeEditor = shallowRef()
const data = reactive({
  code: props.modelValue
})

const extensions = [sql(), oneDark]

interface ICodemirrorPayload {
  view: import("@codemirror/view").EditorView
}

const handleReady = (payload: ICodemirrorPayload) => {
  codeEditor.value = payload.view
}

const handleChange = (value: string) => {
  emits('update:modelValue', value)
}

const getInfo = () => {
  return;
}

interface ICodeEditorExpose {
  getInfo(): any;
}
defineExpose<ICodeEditorExpose>({
  getInfo
})
</script>

<template>
  <codemirror v-model="data.code" placeholder="Code goes here..." :style="{ height: '400px' }" :autofocus="true"
    :indent-with-tab="true" :tab-size="2" :extensions="extensions" @ready="handleReady" @change="handleChange" />
</template>
