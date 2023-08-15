<script setup lang="ts">
import { reactive } from 'vue';
import CodeEditor from './CodeEditor.vue'

const data = reactive({
  code: ""
});

const onClickExecute = () => {
  console.log("code=", data.code);
}
</script>

<template>
  <codeEditor v-model="data.code" />
  <button @click="onClickExecute">Execute</button>
</template>
