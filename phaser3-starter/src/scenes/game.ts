import Phaser from "phaser";

export default class GameScene extends Phaser.Scene {
  constructor() {
    super("GameScene");
  }

  preload() {
    this.load.image("hero", "assets/single_rosalia1.png");
  }

  create() {
    const logo = this.add.image(400, 70, "hero");
    //this.add.text(10, 10, 'Press any button and see the console', {fontSize: '20px'});
  }
}
