<template>
  <div>
    Hello
    <PhaserView :sceneList="[MainScene]" />
  </div>
</template>

<script setup lang="ts">
import PhaserView from "@/components/phaserView.vue";
import MainScene from "@/models/games/mainScene";
</script>

<style lang="scss" scoped>
</style>