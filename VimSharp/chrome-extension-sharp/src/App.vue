<script setup lang="ts">
import HelloWorld from './components/HelloWorld.vue'
import Excel from './components/Excel.vue'
</script>

<template>
  <div class="app-container">
    <HelloWorld msg="Vite + Vue" />
    <Excel />
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