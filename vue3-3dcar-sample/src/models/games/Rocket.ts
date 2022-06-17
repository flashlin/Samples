import type { IImageConstructor } from "./gameTypes";

export class Rocket extends Phaser.Physics.Arcade.Sprite {
  flamePower: number = 0;
  maxFlamePower: number = 10;
  currentVelocity: number = 0;
  addVelocity: number = 0;
  startTime: number = this.getTime();
  fuel: number = 100;

  constructor(aParams: IImageConstructor) {
    super(aParams.scene, aParams.x, aParams.y, aParams.texture, aParams.frame);
    this.initPhysics();
    this.scene.add.existing(this);
  }

  onFlamePower() {
    if (this.flamePower < this.maxFlamePower && this.getDeltaTime()) {
      this.flamePower += 2;
      return;
    }
  }

  offFlamePower() {
    if (this.flamePower > 0 && this.getDeltaTime()) {
      this.flamePower -= 2;
    }
  }

  getDeltaTime() {
    const delta = this.getTime() - this.startTime;
    if (delta > 100) {
      this.startTime = this.getTime();
      return true;
    }
    return false;
  }

  getTime() {
    let d = new Date();
    return d.getTime();
  }

  updateVelocity() {
    if (this.flamePower == 0) {
      this.currentVelocity += 10;
    }

    if (this.flamePower > 0) {
      this.currentVelocity -= this.flamePower;
    }

    if (this.currentVelocity > 500) {
      this.currentVelocity = 500;
    }
    if (this.currentVelocity < -200) {
      this.currentVelocity = -200;
    }
    // console.log(
    //   `y=${this.body.y} v=${this.currentVelocity}, f=${this.flamePower}`
    // );
    if (this.flamePower == 0 && this.body.y >= 522) {
      this.currentVelocity = 0;
    }
  }

  // private initPhysics() {
  //   //this.body.maxVelocity.set(300);
  // }
  private initPhysics() {
    this.scene.physics.world.enable(this);
    this.setVelocity(0, 300);
    this.setBounce(0, 0.5); //彈回去的比例
    this.setCollideWorldBounds(true);
  }
}
