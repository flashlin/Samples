import car1 from './assets/car1.png';
import { ILine, IPosition } from './drawUtils';

const carWidth = 75;
const carHeight = 117;
const carColor = "blue";
const roadLength = 220;
const roadWidth = 220;
const roadColor = 'red';
const roadMargin = 22;

enum RoadType {
    Vertical,
    Horizontal,
    LeftTopCurve,
    LeftBottomCurve,
    RightTopCurve,
    RightBottomCurve,
}



interface IRoad {
    pos: IPosition;
    render(ctx: CanvasRenderingContext2D, pos: IPosition): void;
}

//垂直
class VerticalRoad implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.moveTo(x + roadMargin, y);
        ctx.lineTo(x + roadMargin, y + roadLength);
        ctx.moveTo(x + roadWidth - roadMargin, y);
        ctx.lineTo(x + roadWidth - roadMargin, y + roadLength);
        ctx.lineWidth = 7;
        ctx.strokeStyle = roadColor;  // 線條顏色
        ctx.stroke();
    }
}


// 水平
class HorizontalRoad implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.moveTo(x, y + roadMargin);
        ctx.lineTo(x + roadLength, y + roadMargin);
        ctx.moveTo(x, y + roadWidth - roadMargin);
        ctx.lineTo(x + roadLength, y + roadWidth - roadMargin);
        ctx.lineWidth = 7;
        ctx.strokeStyle = roadColor;
        ctx.stroke();
    }
}


class LeftTopCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x + roadLength,
            y: pos.y + roadWidth,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, roadWidth - roadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + roadMargin - roadMargin, y, roadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }
}

function create2dArray<T>(width: number, height: number): T[][] {
    let arr: T[][] = new Array(width);
    for (let i = 0; i < arr.length; i++) {
        arr[i] = new Array(height);
    }
    return arr;
}

class Line {
    line: ILine;
    color: string = roadColor;

    constructor(line: ILine) {
        this.line = line;
    }
    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const line = this.line;
        ctx.beginPath();
        ctx.moveTo(line.start.x + pos.x, line.start.y + pos.y);
        ctx.lineTo(line.end.x + pos.x, line.end.y + pos.y);
        ctx.lineWidth = 7;
        ctx.strokeStyle = this.color;
        ctx.stroke();
    }
}

class RoadMap {
    pos: IPosition;
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);


    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = pos;
        const curve = new LeftTopCurve()
        this.roads[0][0] = curve;
        this.roads[0][1] = new VerticalRoad();
        this.roads[1][0] = new HorizontalRoad();
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                if (road == null) {
                    continue;
                }
                const roadPos = {
                    x: x + ix * roadWidth,
                    y: y + iy * roadLength,
                };
                road.render(ctx, roadPos);
            }
        }
    }
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