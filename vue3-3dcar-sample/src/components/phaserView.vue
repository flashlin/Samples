<template>
  <div ref="game">
  </div>
</template>

<script setup lang="ts">
import Phaser from 'phaser';
import { onMounted, onUnmounted, ref } from 'vue';

const gameRef = ref<HTMLElement>();

interface PhaserViewProps {
  //modelValue: string;
  sceneList: Phaser.Scene[]
}
const props = defineProps<PhaserViewProps>();

const config = {
  type: Phaser.AUTO,
  parent: gameRef.value,
  width: 800,
  height: 600,
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
  },
  physics: {
    default: 'arcade',
    arcade: {
      gravity: { y: 300 },
    },
  },
  scene: props.sceneList,
} as Phaser.Types.Core.GameConfig;

class Game extends Phaser.Game {
  constructor() {
    super(config);
  }
}

let game: Game;
onMounted(() => {
  game = new Game();
});

onUnmounted(() => {
  game.destroy(true);
});
</script>
