import { Collision, GameObj, KaboomCtx, LifespanCompOpt, Vec2 } from "kaboom";
import { createMap } from "./map1";

const MOVE_SPEED = 100;
export class SceneState {
  constructor(aParams: Partial<SceneState>) {
    Object.assign(this, aParams);
  }
  level: number = 0;
}

export function createScene(k: KaboomCtx) {
  k.scene("Scene1", (state: SceneState) => {
    layers(["bg", "obj", "ui"], "obj");

    const mapsLength = createMap(state.level);

    k.add([
      text("0"),
      pos(400, 450),
      layer("ui"),
      {
        value: 123,
      },
    ]);

    let n = 1;
    for (let x = 0; x < 30; x++) {
      for (let y = 0; y < 30; y++) {
        k.add([
          rect(32, 32),
          pos(x * 30, y * 30),
          area(),
          color(0, 0, 0),
          lifespan(0.1 * n, { fade: 0.5 }),
          //lifespan(0.1 * n),
        ]);
      }
      n++;
    }

    //const player = level.spawn('p', 1, 10)
    let currCam = camPos();

    const player = k.get("player")[0];

    player.onUpdate(() => {
      camPos(vec2(player.pos.x, currCam.y));
    });

    player.onCollide("door", (door, col: Collision) => {
      if (!col.isBottom()) {
        return;
      }
      state.level++;
      if (state.level >= mapsLength) {
        state.level = 0;
      }
      go("Scene1", state);
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

    onKeyDown("down", () => {
      player.move(0, MOVE_SPEED);
    });
  });
}
