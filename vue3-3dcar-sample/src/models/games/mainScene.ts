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
}

interface IScreen {
  width: number;
  height: number;
}

function project3D(
  camera: ICamera,
  worldPoint: IVector3D,
  screen: IScreen
) : IVector2D {
  const transX = worldPoint.x - camera.x;
  const transY = worldPoint.y - camera.y;
  const transZ = worldPoint.z - camera.z;

  const scale = camera.z / transZ;
  const projectedX = scale * transX;
  const projectedY = scale * transY;
  //const projectedZ = scale * transZ;

  return {
    x: projectedX,
    y: projectedY,
  }
}

function project3DRect(camera: ICamera, rect: IVector3D[], screen: IScreen) {
  return rect.map(x => project3D(camera, x, screen));
}

function addRect(point: IVector3D, width: number, height: number) {
  const p1 = { x: point.x, y: point.y, z:point.z };
  const p2 = { x: point.x + width, y: point.y, z:point.z };
  const p3 = { x: point.x + width, y: point.y, z:point.z + height };
  const p4 = { x: point.x, y: point.y, z:point.z + height };
  return [p1, p2, p3, p4];
}

function add3DRect(camera: ICamera, point: IVector3D, width: number, height: number, screen: IScreen) {
  const r1 = addRect(point, width, height);
  const projectedr1 = project3DRect(camera, r1, screen);
  return projectedr1;
}


function renderPolygon(graphics, rect: IVector2D[], color) {
  let polygon = new Phaser.Geom.Polygon(rect);
  graphics.fillStyle(color, 1);
  graphics.fillPoints(polygon.points, true);
}

function renderPolygon1(graphics, p1: IVector2D, p2: IVector2D, p3: IVector2D, p4: IVector2D, color) {
  let polygon = new Phaser.Geom.Polygon([p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y]);
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
  transform: Phaser.Math.Matrix4;

  constructor() {
    super({
      key: "MainScene",
    });
  }

  preload() {
    this.load.image("logo", "assets/game/phaser-logo.png");

    this.load.atlas('atlas', 'assets/game/car.png', 'assets/game/car.json');
  }

  camera: any;

  create() {
    //this.add.image(400, 300, 'logo');
    const graphics = this.add.graphics();

    const car1 = this.add.sprite(200, 200, 'atlas', 0);
    car1.setFlip(true, false);
    car1.setScale(0.3);
  }

  update() {
    //this.camera.transformChildren(this.transform);
  }
}
