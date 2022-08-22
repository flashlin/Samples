import Phaser, { Scene } from "phaser";
import { SceneEffect } from "@/utils/index";

function fadeOutSceneFn(
  scene: Phaser.Scene,
  duration: number,
  nextScene: string
) {
  let running = false;
  return () => {
    if (running) return;
    running = true;
    // scene.cameras.main.fade(duration, 0, 0, 0, false, function() {
    //   scene.scene.start("LandingScene");
    // });
    scene.cameras.main.once(
      Phaser.Cameras.Scene2D.Events.FADE_OUT_COMPLETE,
      (
        cam: Phaser.Cameras.Scene2D.Camera,
        effect: Phaser.Cameras.Scene2D.Effects.Fade
      ) => {
        scene.game.scene.start(nextScene);
      }
    );
    scene.cameras.main.fadeOut(duration, 0, 0, 0);
  };
}

export default class LandingScene extends Phaser.Scene {
  constructor() {
    super("LandingScene");
  }

  preload() {
    this.load.image("logo", "assets/phaser3-logo.png");
  }

  create() {
    const self = this;
    //this.cameras.main.setBackgroundColor("#421278");
    const logo = this.add.image(400, 70, "logo");

    this.tweens.add({
      targets: logo,
      y: 350,
      duration: 1500,
      ease: "Sine.inOut",
      yoyo: true,
      repeat: -1,
    });

    let fadeOut = fadeOutSceneFn(this, 1000, "GameScene");
    let effect = new SceneEffect(this);
    this.input.keyboard.on("keydown-ENTER", (event: KeyboardEvent) => {
      effect.fadeOut(1000, "GameScene");
      //fadeOut();
      //self.scene.start("GameScene");
    });
  }
}

