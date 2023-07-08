import { Car } from './car';
import { ILine, IPosition, drawRect, drawText, posInfo } from './drawUtils';
import { CanvasHeight, CanvasWidth, HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, CarFrameMargin, CarHeight, CarWidth, RoadLength, FPS, CenterX, CenterY, CarPos } from './gameUtils';

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
        const car = this.car;
        car.pos = CarPos;
        car.render(ctx);
        car.drawFrame(ctx);
        const [pos0, pos1] = car.move();
        
        const roadMap = this.roadMap;
        const carBound = this.car.getBound();
        const road = roadMap.collide(ctx, carBound);
        if (road != null) {
            findTwoLinesIntersection({ start: pos0, end: pos1}, );
            road.renderDamaged(ctx);
        }
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = this.roadMap;
        roadMap.pos = { x: - this.car.x + CenterX, y: - this.car.y + CenterY };
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