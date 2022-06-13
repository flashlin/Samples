<template>
  <div>
    Hello
    <PhaserView :sceneList="[MainScene]" />
  </div>
</template>

<script setup lang="ts">
import PhaserView from "@/components/phaserView.vue";
import { Scene } from "phaser";

class MainScene extends Scene {
  constructor() {
    super({
      key: 'MainScene'
    });
  }

  preload() {
    this.load.image('logo', 'assets/game/phaser-logo.png');
  }

  create() {
    this.add.image(400, 300, 'logo');
  }
}



</script>

<style lang="scss" scoped>
</style>