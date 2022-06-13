<template>
   <div ref="game">
   </div>
</template>

<script setup lang="ts">
import Phaser from 'phaser';
import { onMounted, onUnmounted } from 'vue';

const config = {
   type: Phaser.AUTO,
   //parent: 'game',
   width: 800,
   height: 600,
   scene: {
      preload: preload,
      create: create
   }
};

class Game extends Phaser.Game {
  constructor () {
    super(config)
  }
}

function preload ()
{
   //this.load.setBaseURL('http://127.0.0.1');
   this.load.image('logo', 'assets/game/phaser-logo.png');
}

function create ()
{
   this.add.image(400, 300, 'logo');
}

let game: Game;
onMounted(() =>{
   game = new Game();
})

onUnmounted(()=>{
   game.destroy(true);
})
</script>
