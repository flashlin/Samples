<template>
  <div class="loading-overlay">
    <div class="spinner"></div>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
/* Overlay for dark mode */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(20, 20, 20, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

/* Spinner animation */
.spinner {
  width: 64px;
  height: 64px;
  border: 8px solid #333;
  border-top: 8px solid #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style> 