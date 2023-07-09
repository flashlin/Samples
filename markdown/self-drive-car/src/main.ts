import { Car } from './car';
import { drawLine, drawRect, drawText, posInfo } from './drawUtils';
import { CanvasHeight, CanvasWidth, HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, CarFrameMargin, CarHeight, CarWidth, RoadLength, FPS, CenterX, CenterY, CarPos, StartX, StartY } from './gameUtils';
import { rectangleIntersectLine } from './math';

class Game {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    car = new Car();
    roadMap = new RoadMap();
    fps = new FPS();

    constructor() {
        this.canvas = this.createCanvas();
        this.ctx = this.canvas.getContext("2d")!;
        this.car.x = StartX;
        this.car.y = StartY;
    }

    createCanvas() {
        //const canvas = document.createElement("canvas");
        //document.body.appendChild(canvas);
        const canvas = document.getElementById("canvas") as HTMLCanvasElement;
        canvas.width = CanvasWidth;
        canvas.height = CanvasHeight;
        return canvas;
    }

    drawF4Car() {
        const ctx = this.ctx;
        const car = this.car;
        car.pos = CarPos;
        car.render(ctx);
        car.drawFrame(ctx);

        const carPos0 = { x: car.x, y: car.y };
        const carAngle0 = car.angle;
        car.move();
        const carBound1 = car.getBound();

        const roadMap = this.roadMap;
        const [road, collidePoints] = roadMap.collide(ctx, carBound1);
        if (collidePoints.length > 0) {
            car.x = carPos0.x;
            car.y = carPos0.y;
            car.angle = carAngle0;
            road.renderDamaged(ctx);
        }
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = this.roadMap;
        roadMap.pos = { x: CanvasWidth / 2 - this.car.x, y: CanvasHeight / 2 - this.car.y };
        roadMap.render(ctx);
    }

    render() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.drawRoad();
        this.drawF4Car();

        drawLine(ctx, {
            start: { x: 0, y: CanvasHeight / 2 },
            end: { x: CanvasWidth, y: CanvasHeight / 2 }
        }, { strokeSyle: "gray", lineWidth: 3 });
        drawLine(ctx, {
            start: { x: CanvasWidth / 2, y: 0 },
            end: { x: CanvasWidth / 2, y: CanvasHeight }
        }, { strokeSyle: "gray", lineWidth: 3 });

        this.fps.render(ctx, { x: 0, y: 0 });
        requestAnimationFrame(this.render.bind(this));
    }
}

function main() {
    console.log('start...');
    const game = new Game();
    game.render();
}

main();