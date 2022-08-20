import { KaboomCtx } from "kaboom";
import { createMap } from "./map1";

export abstract class BaseScene {
  constructor() {}

  abstract name: string;

  load(k: KaboomCtx) {
    k.scene(this.name, () => {
      this.create(k);
    });
  }

  abstract create(k: KaboomCtx);
}

const MOVE_SPEED = 100;

export class Scene1 extends BaseScene {
  constructor() {
    super();
  }

  name = "Scene1";
  level = 0;

  create(k: KaboomCtx) {
    layers(["bg", "obj", "ui"], "obj");

    createMap(this.level);

    k.add([
      text("0"),
      pos(400, 450),
      layer("ui"),
      {
        value: 123,
      },
    ]);

    //k.add([sprite("mario"), pos(80, 40), scale(2)]);

    //const player = level.spawn('p', 1, 10)
    let currCam = camPos();

    const player = add([
      sprite("mario"),
      area(),
      scale(1),
      //body(),
      pos(80, 40),
      {
        //origin: 'bot',
        dir: vec2(1, 0),
      },
      //lifespan(1, { fade: 0.5}),
    ]);

    player.onUpdate(() => {
      camPos(vec2(player.pos.x, currCam.y));
      //player.resolve();
    });

    keyDown("left", () => {
      //player.use(sprite('abc2'));
      player.move(-MOVE_SPEED, 0);
      player.flipX(true);
    });

    keyDown("right", () => {
      player.move(MOVE_SPEED, 0);
      player.flipX(false);
    });

    keyDown("up", () => {
      player.move(0, -MOVE_SPEED);
    });

    keyDown("down", () => {
      //player.changeSprite("mario_left");
      player.move(0, MOVE_SPEED);
    });
  }
}
