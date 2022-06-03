<template>
  <div class="container">
    <Toast />
    Vue3 Sample
    <router-view></router-view>
  </div>
</template>

<script setup lang="ts">
import { defineComponent, onMounted, reactive, ref } from "vue";
import Toast, { ToastMessageOptions } from "primevue/toast";
import { useToast } from "primevue/usetoast";
import { emitter } from "@/models/AppToast";


onMounted(() => {
  const toast = useToast();
  emitter.on("toast", (data) => {
    toast.add(data as ToastMessageOptions);
  });
});
</script>

<style lang="scss">
</style>