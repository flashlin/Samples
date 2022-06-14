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

    this.load.scenePlugin({
      key: 'Camera3DPlugin',
      url: 'plugins/camera3d.min.js',
      sceneKey: 'cameras3d'
    });

    this.load.image('particle', 'assets/game/blog.png');
  }

  camera: any;

  create() {
    //this.add.image(400, 300, 'logo');
    const graphics = this.add.graphics();
    const roadWidth = 200;
    const roadHeight = 50;
    const roadX = 100;
    const roadY = 100;
    const p1 = { x:roadX, y:roadY, z:0 };
    const p2 =  { x:roadX + roadWidth, y:roadY, z:0 };
    const p3 = { x:roadX + roadWidth, y:roadY + roadHeight, z:20 };
    const p4 = { x:roadX, y: roadY + roadHeight, z: 20};
    renderPolygon(graphics, [p1,p2,p3,p4], "0x397d46");

    const screen = {
      width: 1024,
      height: 800
    };

    const camera: ICamera = {
      x: 400, //screen.width / 2,
      y: 600, //screen.height / 2,
      z: -400,
    };
    const d1 = project3D(camera, p1, screen);
    const d2 = project3D(camera, p2, screen);
    const d3 = project3D(camera, p3, screen);
    const d4 = project3D(camera, p4, screen);

    console.table(d1)

    renderPolygon(graphics, [d1, d2, d3, d4], "0x390046");


    const rectList = function* () {
      yield { x: 100, y: 120, z: 0 }
      yield { x: 100, y: 120, z: 100 }
      yield { x: 100, y: 120, z: 200 }
      yield { x: 100, y: 120, z: 300 }
    }

    for (var item of rectList()) {
      const r = add3DRect(camera, item, roadWidth, roadHeight, screen);
      renderPolygon(graphics, r, "0x595959");
    }


    const rectList2 = function* () {
      yield { x: -200, y: 120, z: 0 }
      yield { x: -200, y: 120, z: 100 }
      yield { x: -200, y: 120, z: 200 }
      yield { x: -200, y: 120, z: 300 }
    }
    for (var item of rectList2()) {
      const r = add3DRect(camera, item, roadWidth, roadHeight, screen);
      renderPolygon(graphics, r, "0x095909");
    }
  }

  update() {
    //this.camera.transformChildren(this.transform);
  }
}
