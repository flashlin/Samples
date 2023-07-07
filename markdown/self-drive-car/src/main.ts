import { Car } from './car';
import { ILine, IPosition, drawRect } from './drawUtils';
import { CanvasHeight, CanvasWidth, HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, CarFrameMargin, CarHeight, CarWidth, RoadLength, FPS } from './gameUtils';


class Game {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    car = new Car();
    roadMap = new RoadMap();
    fps = new FPS();

    constructor() {
        this.canvas = this.createCanvas();
        this.ctx = this.canvas.getContext("2d")!;
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
        const pos = { x: CanvasWidth / 2 - 32, y: CanvasHeight / 2 - 56 };
        this.car.pos = pos;
        this.car.render(ctx);
        this.car.move();
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = this.roadMap;
        // let x = CanvasWidth / 2 - 90;
        // let y = CanvasHeight / 2 - 180;
        // y -= this.car.y;
        // x -= this.car.x;
        // const pos = { x, y };
        const CenterX = CanvasWidth / 2 - CarWidth / 2;
        const CenterY = CanvasHeight / 2 - CanvasHeight / 2;
        const drawPos = { x: CanvasWidth / 2 - CarWidth / 2, y: CanvasHeight / CarHeight / 2 };
        roadMap.pos = { x: this.car.x + CenterX, y: this.car.y + CenterY };
        roadMap.render(ctx);
        const carBound = this.car.getBound();
        drawRect(ctx, carBound, { lineWidth: 5, strokeSyle: "yellow" });
        const road = roadMap.collide(carBound);
        if (road != null) {
            console.log('damaged', drawPos);
            road.renderDamaged(ctx, drawPos);
        }
    }

    render() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.drawRoad();
        this.drawF4Car();


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