import Phaser from 'phaser';
import config from './config';
import LandingScene from './scenes/landing';
import GameScene from './scenes/game';

new Phaser.Game(
  Object.assign(config, {
    scene: [LandingScene, GameScene]
  })
);