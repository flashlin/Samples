import { Scene } from "phaser";
import { Flame } from "./Flame";
import { Rocket } from "./Rocket";

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

function renderPolygon(graphics: Phaser.GameObjects.Graphics, rect: IVector2D[], color: number) {
  let polygon = new Phaser.Geom.Polygon(rect);
  graphics.fillStyle(color, 1);
  graphics.fillPoints(polygon.points, true);
}

function renderPolygon1(
  graphics: Phaser.GameObjects.Graphics,
  p1: IVector2D,
  p2: IVector2D,
  p3: IVector2D,
  p4: IVector2D,
  color: number
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

export default class MainScene extends Scene {
  upArrow!: Phaser.Input.Keyboard.Key;
  downArrow!: Phaser.Input.Keyboard.Key;
  leftArrow!: Phaser.Input.Keyboard.Key;
  rightArrow!: Phaser.Input.Keyboard.Key;
  rocket!: Rocket;
  flame!: Flame;
  startTime!: number;

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
    this.load.image("fireball", "assets/game/fireball.png");
    this.load.image("ground", "assets/game/ground1.png");
  }

  create() {
    this.startTime = this.getTime();

    const ground = this.physics.add.staticGroup({
      key: "ground",
      repeat: 10,
      setXY: { x: 0, y: screen.height-500, stepX: 100 }
    });
    ground.children.iterate((child) => {
      const item = child as Phaser.Physics.Arcade.Image;
      item.setScale(0.05);
    });
    ground.refresh();

    //this.add.image(400, 300, 'logo');
    const text = this.add.text(10, 10, "fuel", {fontSize: "16px"});

    const flame = this.flame = new Flame({
      scene: this,
      x: 200,
      y: 220,
      texture: "flame",
    });
    flame.setScale(0.15);
    flame.rotation = -0.78;

    const rocket = this.rocket = new Rocket({
      scene: this,
      x: 200,
      y: 200,
      texture: "rocket1",
    });
    //car.setFlip(true, false);
    rocket.setScale(0.15);
    rocket.rotation = -0.8;

    this.cameras.main.setBounds(0, 0, screen.width, screen.height);
    //this.cameras.main.startFollow(rocket, true, 0.5, 0.5 ); 
    this.cameras.main.startFollow(rocket); 

    this.physics.add.collider(this.rocket, ground);

    const fireballs = this.physics.add.group({
      key: "fireball",
      repeat: 5,
      setXY: { x: 400, y: 0, stepX: 100 },
    });
    fireballs.children.iterate((child) => {
      //const item = child as Phaser.GameObjects.Image;
      const item = child as Phaser.Physics.Arcade.Image;
      item.setScale(0.1);
      item.body.velocity.x = -Phaser.Math.FloatBetween(10, 300);
      item.body.velocity.y = Phaser.Math.FloatBetween(10, 100);
      //(item as any).setBounceY(Phaser.Math.FloatBetween(0.4, 0.8));
      //item.setBounce(0.5, 0.5);
    });

    this.physics.add.overlap(this.rocket, fireballs, this.collectFireball);

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
    //   this.car.updateFlamePower(true);
    //   this.flame.visible = true;
    // });
  }

  collectFireball(rocket: any, fireball: any) {
    fireball.disableBody(true, true);
    //rocket.flamePower += 10;
  }

  update() {
    //this.car.body.angularVelocity = 0;
    this.flame.visible = false;
    if (this.leftArrow.isDown) {
      this.rocket.x -= 2;
    }
    if (this.rightArrow.isDown) {
      //this.physics.moveToObject(this.car, {x: this.car.x+2, y: this.car.y}, 1);
      this.rocket.x += 2;
    }

    if (this.upArrow.isDown) {
      this.rocket.onFlamePower();
      this.flame.visible = true;
    } else {
      this.rocket.offFlamePower();
    }

    if (this.downArrow.isDown) {
      //this.car.body.y += 2;
    }

    if (this.rocket.flamePower > 0) {
      //this.car.body.velocity.y -= this.car.flamePower;
      this.flame.visible = true;
    }

    this.rocket.updateVelocity();
    this.rocket.body.velocity.y = this.rocket.currentVelocity;
    this.flame.x = this.rocket.x;
    this.flame.y = this.rocket.y + (43 / 10) * this.rocket.flamePower;
  }

  getTime() {
    let d = new Date();
    return d.getTime();
  }
}
