import car1 from './assets/car1.png';
import { ILine, IPosition } from './drawUtils';
import { HorizontalRoad, IRoad, LeftTopCurve, RoadMap, VerticalRoad, carHeight, carWidth, roadLength } from './gameUtils';

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

    constructor(pos: IPosition) {
        this.pos = pos;
        this.carImage = new Image();
        this.carImage.src = car1;
    }

    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        //ctx.globalCompositeOperation = 'destination-atop';
        ctx.drawImage(this.carImage,
            pos.x, pos.y, carWidth, carHeight);
    }
}

class Game {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;

    constructor() {
        this.canvas = this.createCanvas();
        this.ctx = this.canvas.getContext("2d")!;
    }

    createCanvas() {
        //const canvas = document.createElement("canvas");
        //document.body.appendChild(canvas);
        const canvas = document.getElementById("canvas") as HTMLCanvasElement;
        canvas.width = 800;
        canvas.height = 600;
        return canvas;
    }

    drawF4Car() {
        const ctx = this.ctx;
        const car = new Car({ x: 23 + carWidth, y: roadLength });
        car.render(ctx);

        const car2 = new Car({ x: 23, y: roadLength });
        car2.render(ctx);
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