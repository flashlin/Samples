import car1 from './assets/car1.png';
import { ILine, IPosition } from './drawUtils';
import { HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, carFrameMargin, carHeight, carWidth, roadLength } from './gameUtils';

enum RoadType {
    Vertical,
    Horizontal,
    LeftTopCurve,
    LeftBottomCurve,
    RightTopCurve,
    RightBottomCurve,
}

class Car {
    carImage: HTMLImageElement;
    pos: IPosition;
    frame: Rectangle;

    constructor(pos: IPosition) {
        this.pos = pos;
        this.carImage = new Image();
        this.carImage.src = car1;
        this.frame = new Rectangle({
            x: pos.x + carFrameMargin, 
            y: pos.y + carFrameMargin}, 
            carWidth - carFrameMargin * 2, 
            carHeight - carFrameMargin * 2);
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        this.frame.render(ctx, pos);
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        //ctx.globalCompositeOperation = 'destination-atop';
        ctx.drawImage(this.carImage, x, y, carWidth, carHeight);
    }
}

class Game {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    car = new Car({ x: 90, y: 90 });

    constructor() {
        this.canvas = this.createCanvas();
        this.ctx = this.canvas.getContext("2d")!;
    }

    createCanvas() {
        //const canvas = document.createElement("canvas");
        //document.body.appendChild(canvas);
        const canvas = document.getElementById("canvas") as HTMLCanvasElement;
        canvas.width = 800;
        canvas.height = 700;
        return canvas;
    }

    drawF4Car() {
        const ctx = this.ctx;
        const pos = {x: 0, y: 0};
        this.car.render(ctx, pos);
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = new RoadMap();
        roadMap.render(ctx, {x: 0, y: 0});
    }

    render() {
        const ctx = this.ctx;
        const canvas = this.canvas;
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.drawRoad();

        this.drawF4Car();
        requestAnimationFrame(this.render.bind(this));
    }
}

function main() {
    console.log('start...');
    const game = new Game();
    game.render();
}

main();