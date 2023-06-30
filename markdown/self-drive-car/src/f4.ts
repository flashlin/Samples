import car1 from './assets/car1.png';

const carWidth = 75;
const carHeight = 117;
const carColor = "blue";
const roadLength = 220;
const roadWidth = 220;
const roadColor = 'red';
const roadMargin = 22;


type IArc = {
    x: number,
    y: number,
    radius: number,
    startAngle: number,
    endAngle: number
};

type ILine = {
    x1: number,
    y1: number,
    x2: number,
    y2: number
};

enum RoadType {
    Vertical,
    Horizontal,
    LeftTopCurve,
    LeftBottomCurve,
    RightTopCurve,
    RightBottomCurve,
}


interface Position {
    x: number;
    y: number;
}

function computeIntersectXY(arc: IArc, line: ILine): Position | null {
    // Line's equations
    const m = (line.y2 - line.y1) / (line.x2 - line.x1); // slope
    const b = line.y1 - m * line.x1; // y-intercept

    const A = 1 + m * m;
    const B = -2 * arc.x + 2 * b * m - 2 * arc.y * m;
    const C = arc.x * arc.x + b * b - 2 * arc.y * b + arc.y * arc.y - arc.radius * arc.radius;

    // Solve quadratic equation for x
    const det = B * B - 4 * A * C;

    if (det < 0) {
        return null; // no solution
    }

    const x1 = (-B + Math.sqrt(det)) / (2 * A);
    const x2 = (-B - Math.sqrt(det)) / (2 * A);

    // Calculate corresponding y values
    const y1 = m * x1 + b;
    const y2 = m * x2 + b;

    // Check if intersection points are within line segment and the arc
    const intersections = [{ x: x1, y: y1 }, { x: x2, y: y2 }];

    for (let point of intersections) {
        const angle = Math.atan2(point.y - arc.y, point.x - arc.x);
        if (point.x >= Math.min(line.x1, line.x2) && point.x <= Math.max(line.x1, line.x2) &&
            angle >= arc.startAngle && angle <= arc.endAngle) {
            return point;
        }
    }

    return null;
}

interface IRoad {
    pos: Position;
    render(ctx: CanvasRenderingContext2D): void;
}

//垂直
class VerticalRoad implements IRoad {
    pos: Position;

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        ctx.beginPath();
        ctx.moveTo(pos.x + roadMargin, pos.y);
        ctx.lineTo(pos.x + roadMargin, pos.y + roadLength);
        ctx.moveTo(pos.x + roadWidth - roadMargin, pos.y);
        ctx.lineTo(pos.x + roadWidth - roadMargin, pos.y + roadLength);
        ctx.lineWidth = 7;
        ctx.strokeStyle = roadColor;  // 線條顏色
        ctx.stroke();
    }
}


// 水平
class HorizontalRoad implements IRoad {
    pos: Position;

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y + roadMargin);
        ctx.lineTo(pos.x + roadLength, pos.y + roadMargin);
        ctx.moveTo(pos.x, pos.y + roadWidth - roadMargin);
        ctx.lineTo(pos.x + roadLength, pos.y + roadWidth - roadMargin);
        ctx.lineWidth = 7;
        ctx.strokeStyle = roadColor;
        ctx.stroke();
    }
}


class LeftTopCurve implements IRoad {
    pos: Position;

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        pos.x += roadLength;
        pos.y += roadWidth;
        ctx.beginPath();
        //ctx.arc(pos.x, pos.y, roadWidth - roadMargin, Math.PI, 1.5 * Math.PI);
        ctx.arc(pos.x, pos.y, roadWidth - roadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(pos.x + roadMargin - roadMargin, pos.y, roadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    getArc(): IArc {
        const pos = this.pos;
        pos.x += roadLength;
        pos.y += roadWidth;
        return {
            x: pos.x + roadWidth - roadMargin,
            y: pos.y,
            radius: roadWidth - roadMargin,
            startAngle: Math.PI,
            endAngle: 1.5 * Math.PI
        };
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
    constructor(line: ILine) {
        this.line = line;
    }
    render(ctx: CanvasRenderingContext2D) {
        const line = this.line;
        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.lineWidth = 7;
        ctx.strokeStyle = roadColor;  // 線條顏色
        ctx.stroke();
    }
}

class RoadMap {
    pos: Position;
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
        const curve = new LeftTopCurve()
        this.roads[0][0] = curve;
        this.roads[0][1] = new VerticalRoad();
        this.roads[1][0] = new HorizontalRoad();

        const arc = curve.getArc();
        const line = { x1: 0, y1: 0, x2: 100, y2: 100 };
        const rc = computeIntersectXY(arc, line);
        if (rc == null) {
            console.log('no!!');
        } else {
            console.log(`rc = ${rc.x} ${rc.y}`);
        }
    }

    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                if (road == null) {
                    continue;
                }
                road.pos.x = pos.x + ix * roadWidth;
                road.pos.y = pos.y + iy * roadLength;
                road.render(ctx);
            }
        }

        const l = new Line({ x1: 0, y1: 0, x2: 100, y2: 100 });
        l.render(ctx);
    }
}

class Car {
    carImage: HTMLImageElement;
    pos: Position;

    constructor(pos: Position) {
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
        roadMap.render(ctx);

        // const road = new HorizontalRoad({ x: 3 + roadWidth, y: 0 });
        // road.render(ctx);

        // const road1 = new LeftTopCurve({ x: 0, y: 0 });
        // road1.render(ctx);

        // const road2 = new VerticalRoad({ x: 0, y: roadLength });
        // road2.render(ctx);
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