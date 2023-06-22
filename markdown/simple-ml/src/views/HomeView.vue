<script setup lang="ts">
</script>

<template>
  <div class="container">
    <div class="title">人工智慧</div>
    <img src="@/assets/images/AI.jpg">
  </div>
</template>

<style scoped>
.container {
  position: relative;
  text-align: center;
}

.title {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 8em;
  font-weight: bold; 
  background: linear-gradient(to right, blue, purple);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>