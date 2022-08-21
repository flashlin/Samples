import Phaser from 'phaser';

export default class LandingScene extends Phaser.Scene {
  constructor() {
    super('LandingScene');
  }

  preload() {
    this.load.image('logo', 'assets/phaser3-logo.png');
  }

  create() {
    const self = this;
    const logo = this.add.image(400, 70, 'logo');

    this.tweens.add({
      targets: logo,
      y: 350,
      duration: 1500,
      ease: 'Sine.inOut',
      yoyo: true,
      repeat: -1
    });


    this.input.keyboard.on('keydown-ENTER', (event: KeyboardEvent) => {
      self.scene.start('GameScene');
    });
  }
}