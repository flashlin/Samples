import type { IImageConstructor } from "./gameTypes";

export class Flame extends Phaser.GameObjects.Image {
  constructor(aParams: IImageConstructor) {
    super(aParams.scene, aParams.x, aParams.y, aParams.texture, aParams.frame);
    this.scene.add.existing(this);
  }
}
