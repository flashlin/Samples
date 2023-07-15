import { Car } from './car';
import { drawLine, drawRect, drawText, posInfo } from './drawUtils';
import { CanvasHeight, CanvasWidth, HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, CarFrameMargin, CarHeight, CarWidth, FPS, CenterX, CenterY, CarPos, StartX, StartY, sleepNow, UseBrain } from './gameUtils';
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

    async drawF4Car() {
        const roadMap = this.roadMap;
        const ctx = this.ctx;
        const car = this.car;
        car.pos = CarPos;
        car.render(ctx);
        car.drawFrame(ctx);
        await car.move(ctx, roadMap);
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = this.roadMap;
        roadMap.pos = { x: CanvasWidth / 2 - this.car.x, y: CanvasHeight / 2 - this.car.y };
        roadMap.render(ctx);
    }

    async render() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.drawRoad();
        await this.drawF4Car();

        //十字標記
        // drawLine(ctx, {
        //     start: { x: 0, y: CanvasHeight / 2 },
        //     end: { x: CanvasWidth, y: CanvasHeight / 2 }
        // }, { strokeSyle: "gray", lineWidth: 3 });
        // drawLine(ctx, {
        //     start: { x: CanvasWidth / 2, y: 0 },
        //     end: { x: CanvasWidth / 2, y: CanvasHeight }
        // }, { strokeSyle: "gray", lineWidth: 3 });

        this.fps.render(ctx, { x: 0, y: 0 });
        //console.log('render...')
        requestAnimationFrame(this.render.bind(this));
    }
}

function main() {
    console.log('start...');
    const game = new Game();
    game.render();
}

main();