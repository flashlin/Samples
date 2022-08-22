import Phaser from "phaser";

export class SceneEffect {
  scene: Phaser.Scene;
  running = false;

  constructor(scene: Phaser.Scene) {
    this.scene = scene;
  }

  fadeOut(duration: number, nextScene: string | Phaser.Scene) {
    if (this.running)
      return;
    let scene = this.scene;
    this.running = true;
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
  }
}
