<script setup lang="ts">
import { StagewiseToolbar } from '@stagewise/toolbar-vue';
import { VisualRouterView } from '@/tools/visual-router'

const stagewiseConfig = {
  plugins: []
};

// 判斷是否為開發模式
const isDev = import.meta.env.MODE === 'development';
</script>

<template>
  <StagewiseToolbar v-if="isDev" :config="stagewiseConfig" />
  <div class="app-container">
    <VisualRouterView />
  </div>
</template>

<style scoped>
.app-container {
  width: 100%;
  height: 100%;
  overflow: auto;
  box-sizing: border-box;
  padding: 1rem;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}

</style>