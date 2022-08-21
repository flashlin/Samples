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

    const mapsLength = createMap(this.level);

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

    const player = k.get("player")[0];

    player.onUpdate(() => {
      camPos(vec2(player.pos.x, currCam.y));
    });

    player.onCollide("door", () => {
      this.level++;
      if (this.level >= mapsLength) {
        this.level = 0;
      }
      go("Scene1", {
        level: this.level,
      });
    });

    onKeyDown("left", () => {
      //player.use(sprite('abc2'));
      player.move(-MOVE_SPEED, 0);
      player.flipX(true);
    });

    onKeyDown("right", () => {
      player.move(MOVE_SPEED, 0);
      player.flipX(false);
    });

    onKeyDown("up", () => {
      player.move(0, -MOVE_SPEED);
    });

    onKeyDown("down", () =>{
      player.move(0, MOVE_SPEED);
    });
  }
}
