<script setup lang="ts">
import SideMenu from '@/components/SideMenu.vue'
</script>

<template>
  <SideMenu v-if="$route.meta.requiresAuth" />
  Main
</template>
