<script setup lang="ts">
import { reactive, shallowRef } from 'vue';
import { Codemirror } from 'vue-codemirror'
import { sql } from '@codemirror/lang-sql'
import { oneDark } from '@codemirror/theme-one-dark'


const codeEditor = shallowRef()
const data = reactive({
  code: ""
})
const extensions = [sql(), oneDark]

interface ICodemirrorPayload {
  view: import("@codemirror/view").EditorView
}

const handleReady = (payload: ICodemirrorPayload) => {
  codeEditor.value = payload.view
}
</script>

<template>
  <codemirror v-model="data.code" 
    placeholder="Code goes here..."
    :style="{ height: '400px'}"
    :autofocus="true"
    :indent-with-tab="true"
    :tab-size="2"
    :extensions="extensions"
    @ready="handleReady"
  />
</template>
