import { Scene1 } from "./scenes/scene1";
import kaboom from "kaboom";
import { MapSize } from "./types";

const k = kaboom({
  debug: true,
  width: 320,
  height: 240,
  scale: 2,
});

const showMap = () => {
  for (let y = 0; y < MapSize.height; y++) {
    for (let x = 0; x < MapSize.width; x++) {
      console.log(`${x},${y}`);
    }
  }
};

loadRoot("assets/");
loadSprite("rock", "rock.png");
loadSprite("door", "door.png");
loadSprite("mario", "mario.png");
loadAseprite("enemies", "enemies.png", "enemies.json");

let scene1 = new Scene1();
scene1.load(k);

go("Scene1");

onClick(() => {
  //addKaboom(mousePos())
});
