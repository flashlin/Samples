import car1 from './assets/car1.png';
import { ILine, IPosition } from './drawUtils';
import { CanvasHeight, CanvasWidth, HorizontalRoad, IRoad, LeftTopCurve, Rectangle, RoadMap, VerticalRoad, carFrameMargin, carHeight, carWidth, roadLength } from './gameUtils';

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

    constructor() {
        this.pos = {
            x: 0,
            y: 0
        };
        this.carImage = new Image();
        this.carImage.src = car1;
        this.frame = new Rectangle({
            x: this.pos.x + carFrameMargin, 
            y: this.pos.y + carFrameMargin}, 
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
    car = new Car();

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
        const pos = {x: CanvasWidth/2-32, y: CanvasHeight/2-56};
        this.car.render(ctx, pos);
    }

    drawRoad() {
        const ctx = this.ctx;
        const roadMap = new RoadMap();
        roadMap.render(ctx, {x: CanvasWidth / 2 - 90, y: CanvasHeight / 2 - 180});
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