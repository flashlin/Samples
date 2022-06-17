<template>
  <div ref="gameView">
  </div>
</template>

<script setup lang="ts">
import Phaser from 'phaser';
import { onMounted, onUnmounted, ref } from 'vue';

const gameView = ref<HTMLElement>();

interface PhaserViewProps {
  //modelValue: string;
  sceneList: Phaser.Scene[]
}
const props = defineProps<PhaserViewProps>();

const config = {
  type: Phaser.AUTO,
  //parent: gameView.value,
  width: 800,
  height: 600,
  fps: {
    target: 30,
    //forceSetTimeOut: true,
  },
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
  constructor(config: Phaser.Types.Core.GameConfig) {
    super(config);
  }
}

let game: Game;
onMounted(() => {
  config.parent = gameView.value;
  game = new Game(config);
});

onUnmounted(() => {
  game.destroy(true);
});
</script>
