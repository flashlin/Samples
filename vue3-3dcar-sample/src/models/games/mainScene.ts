import { Scene } from "phaser";

export default class MainScene extends Scene {
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
