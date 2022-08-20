import { KaboomCtx, LevelOpt } from "kaboom";
import { patrol } from "../models";

const maps = [
  [
    "====================",
    "=                  =",
    "=                  =",
    "=                  =",
    "=                  =",
    "=         E        =",
    "=                  =",
    "=                  =",
    "=                  =",
    "====================",
  ],
  [
   "====================",
   "=  E               =",
   "=                  =",
   "=                  =",
   "=                  =",
   "=                  =",
   "=                  =",
   "=                  =",
   "=                  =",
   "====================",
 ],
];

export function createMap(idx: number = 0) {
  const cfg: LevelOpt = {
    width: 20,
    height: 20,
    pos: vec2(0, 0),
    "=": () => [sprite("rock"), area(), solid(), scale(0.5)],
    E: () => [
      sprite("enemies", { anim: "Walking" }),
      area({ width: 16, height: 16 }),
      patrol(),
      solid(),
      "badGuy",
    ],
  };

  addLevel(maps[idx], cfg);

  //add([sprite("mario"), layer('bg')]);
}
