import { Scene } from "phaser";

interface IVector3D {
  x: number;
  y: number;
  z: number;
}

interface IVector2D {
  x: number;
  y: number;
}

interface ICamera {
  x: number;
  y: number;
  z: number;
  depth: number;
}

interface IScreen {
  width: number;
  height: number;
}

function project3D(
  camera: ICamera,
  worldPoint: IVector3D,
  screen: IScreen
): IVector2D {
  const transX = worldPoint.x - camera.x;
  const transY = worldPoint.y - camera.y;
  //const transZ = worldPoint.z - camera.z;

  const scale = camera.depth / (worldPoint.z - camera.z);
  const projectedX = ((1 + scale * transX) * screen.width) / 2;
  const projectedY = ((1 - scale * transY) * screen.height) / 2;

  return {
    x: projectedX,
    y: projectedY,
  };
}

function project3DRect(camera: ICamera, rect: IVector3D[], screen: IScreen) {
  return rect.map((x) => project3D(camera, x, screen));
}

function addRect(point: IVector3D, width: number, height: number) {
  const p1 = { x: point.x, y: point.y, z: point.z };
  const p2 = { x: point.x + width, y: point.y, z: point.z };
  const p3 = { x: point.x + width, y: point.y, z: point.z + height };
  const p4 = { x: point.x, y: point.y, z: point.z + height };
  return [p1, p2, p3, p4];
}

function add3DRect(
  camera: ICamera,
  point: IVector3D,
  width: number,
  height: number,
  screen: IScreen
) {
  const r1 = addRect(point, width, height);
  const projectedr1 = project3DRect(camera, r1, screen);
  return projectedr1;
}

function renderPolygon(graphics, rect: IVector2D[], color) {
  let polygon = new Phaser.Geom.Polygon(rect);
  graphics.fillStyle(color, 1);
  graphics.fillPoints(polygon.points, true);
}

function renderPolygon1(
  graphics,
  p1: IVector2D,
  p2: IVector2D,
  p3: IVector2D,
  p4: IVector2D,
  color
) {
  let polygon = new Phaser.Geom.Polygon([
    p1.x,
    p1.y,
    p2.x,
    p2.y,
    p3.x,
    p3.y,
    p4.x,
    p4.y,
  ]);
  graphics.fillStyle(color, 1);
  graphics.fillPoints(polygon.points, true);

  // this.scene.graphics.lineStyle(2, 0x9600ff, 1); //opacity - 1
  // this.scene.graphics.beginPath();
  // this.scene.graphics.moveTo(polygon.points[0].x, polygon.points[0].y);
  // for (let i = 1; i < polygon.points.length; i++) {
  //   this.scene.graphics.lineTo(polygon.points[i].x, polygon.points[i].y);
  // }
  // this.scene.graphics.closePath();
  // this.scene.graphics.strokePath();
}

export interface IImageConstructor {
  scene: Phaser.Scene;
  x: number;
  y: number;
  texture: string | Phaser.Textures.Texture;
  frame?: string | number;
}

class Rocket extends Phaser.GameObjects.Image {
  body: Phaser.Physics.Arcade.Body;

  constructor(aParams: IImageConstructor) {
    super(aParams.scene, aParams.x, aParams.y, aParams.texture, aParams.frame);
    this.initPhysics();
    this.scene.add.existing(this);
  }

  private initPhysics() {
    this.scene.physics.world.enable(this);
    this.body.setVelocity(0, 300); 
    //this.body.setBounce(1, 1);
    this.body.setBounce(0, 0.5); //彈回去的比例
    this.body.setCollideWorldBounds(true);
  }
}

class Flame extends Phaser.GameObjects.Image {
  constructor(aParams: IImageConstructor) {
    super(aParams.scene, aParams.x, aParams.y, aParams.texture, aParams.frame);
    this.scene.add.existing(this);
  }
}

export default class MainScene extends Scene {
  platforms : Phaser.Physics.Arcade.StaticGroup;
  upArrow: Phaser.Input.Keyboard.Key;
  downArrow: Phaser.Input.Keyboard.Key;
  leftArrow: Phaser.Input.Keyboard.Key;
  rightArrow: Phaser.Input.Keyboard.Key;
  car: Rocket;//'Phaser.GameObjects.Sprite;
  flame: Flame;

  constructor() {
    super({
      key: "MainScene",
    });
  }

  preload() {
    this.load.image("logo", "assets/game/phaser-logo.png");
    this.load.atlas("atlas", "assets/game/car.png", "assets/game/car.json");
    this.load.image("rocket1", "assets/game/rocket1.png");
    this.load.image("flame", "assets/game/flame.png");
  }

  create() {
    //this.add.image(400, 300, 'logo');
    const graphics = this.add.graphics();

    const flame = this.flame = new Flame({scene:this, x: 200, y: 220, texture: "flame" });
    flame.setScale(0.15);
    flame.rotation = -0.78;

    //const car = (this.car = this.add.sprite(200, 200, "atlas", 0));
    const car = this.car = new Rocket({scene:this, x: 200, y: 200, texture: "rocket1" });
    //car.setFlip(true, false);
    car.setScale(0.15);
    car.rotation = -0.8;
    //car.setGravityY(200);
    //this.physics.world.enable(car, 0);



    this.leftArrow = this.input.keyboard.addKey(
      Phaser.Input.Keyboard.KeyCodes.LEFT
    );
    this.rightArrow = this.input.keyboard.addKey(
      Phaser.Input.Keyboard.KeyCodes.RIGHT
    );
    this.upArrow = this.input.keyboard.addKey(
      Phaser.Input.Keyboard.KeyCodes.UP
    );
    this.downArrow = this.input.keyboard.addKey(
      Phaser.Input.Keyboard.KeyCodes.DOWN
    );
    // leftArrow.on("up", () => {
    //   car1.x += 2;
    // });
  }

  update() {

    this.flame.visible = false;
    if (this.leftArrow.isDown) {
      this.car.body.x -= 2;
    }
    if (this.rightArrow.isDown) {
      //this.physics.moveToObject(this.car, {x: this.car.x+2, y: this.car.y}, 1);
      this.car.body.x += 2;
    }
    if (this.upArrow.isDown) {
      this.car.body.y -= 2;
      this.flame.visible = true;
    }
    if (this.downArrow.isDown) {
      this.car.body.y += 2;
    }
    this.flame.y = this.car.y + 43;
  }
}
