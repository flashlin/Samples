import { KaboomCtx, LevelOpt } from "kaboom";
import { patrol } from "../models";

const maps = [
  [
    "====================",
    "=                  =",
    "=         p        =",
    "=         d        =",
    "=                  =",
    "=         e        =",
    "=                  =",
    "=                  =",
    "=                  =",
    "====================",
  ],
  [
    "====================",
    "=  e               =",
    "=                  =",
    "=                  =",
    "=                  =",
    "=                  =",
    "=                  =",
    "=                  =",
    "= dp               =",
    "====================",
  ],
];

export function createMap(idx: number = 0) {
  const cfg: LevelOpt = {
    width: 20,
    height: 20,
    pos: vec2(0, 0),
    "=": () => [sprite("rock"), area(), solid(), scale(0.5), "wall"],
    d: () => [sprite("door"), area(), scale(0.5), "door"],
    p: () => [
      "player",
      sprite("mario"),
      area(),
      solid(),
      scale(1),
      //body(),
      {
        //origin: 'bot',
        dir: vec2(1, 0),
      },
      //lifespan(1, { fade: 0.5}),
    ],
    e: () => [
      sprite("enemies", { anim: "Walking" }),
      area({ width: 24, height: 24 }),
      solid(),
      patrol(),
      "badGuy",
    ],
  };

  addLevel(maps[idx], cfg);

  //add([sprite("mario"), layer('bg')]);
  return maps.length;
}
