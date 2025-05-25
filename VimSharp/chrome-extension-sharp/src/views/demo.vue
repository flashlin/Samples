<script setup lang="ts">
import { ref } from 'vue'
import VimCodeEditor from '@/components/VimCodeEditor.vue'

const code = ref('Hello World')
</script>
<template>
   <vim-code-editor v-model:value="code" />
</template>