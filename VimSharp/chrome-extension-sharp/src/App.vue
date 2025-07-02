<script setup lang="ts">
import { provide, ref, reactive } from 'vue';
import Loading from '@/components/Loading.vue';
import { ProvideKeys } from '@/tools/ProvideTypes';
// 判斷是否為開發模式
// const isDev = import.meta.env.MODE === 'development';

const loadingState = reactive({
  isLoading: false
});
provide(ProvideKeys.LoadingState, loadingState);
</script>

<template>
  <div class="app-container">
    <Loading v-if="loadingState.isLoading" />
    <RouterView />
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