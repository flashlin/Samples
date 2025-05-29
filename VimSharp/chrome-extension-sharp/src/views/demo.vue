<script setup lang="ts">
import { ref } from 'vue'
import VimCodeEditor from '@/components/vimCodeEditor.vue'

const code = ref('Hello World')
</script>
<template>
  <div style="height: 100vh; padding: 24px; box-sizing: border-box;">
    <VimCodeEditor v-model:value="code" :enable-vim="false" />
  </div>
</template>