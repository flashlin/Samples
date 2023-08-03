<script setup lang="ts">
</script>

<template>
  <n-card title="Chat Train Data">
    <n-space vertical>
      Instruction
      <n-mention type="textarea" />
      Input
      <n-mention type="textarea" />
      Output
      <n-mention type="textarea" />
      <n-space>
        <n-button>Save</n-button>
      </n-space>
    </n-space>
  </n-card>
</template>

<style scoped>
.n-card {
  max-width: 100%;
}
</style>