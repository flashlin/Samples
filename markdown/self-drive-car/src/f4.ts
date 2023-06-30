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

function getArcStartAngleXY(arc: IArc) {
    const pos = { x: arc.x, y: arc.y };
    const radius = roadWidth - roadMargin;
    const startAngle = Math.PI;
    return {
        x: pos.x + radius * Math.cos(startAngle),
        y: pos.y + radius * Math.sin(startAngle)
    };
}

function getArcEndAngleXY(arc: IArc) {
    const pos = { x: arc.x, y: arc.y };
    const radius = roadWidth - roadMargin;
    const endAngle = 1.5 * Math.PI;
    return {
        x: pos.x + radius * Math.cos(endAngle),
        y: pos.y + radius * Math.sin(endAngle),
    };
}

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

//是否垂直
function isPerpendicular(line: ILine, pos: Position) {
    const slope1 = (line.y2 - line.y1) / (line.x2 - line.x1);
    const slope2 = (pos.y - line.y1) / (pos.x - line.x1);
    const product = slope1 * slope2;
    const tolerance = 1e-10;
    return Math.abs(product + 1) < tolerance;
}

function computeCoefficients(line: ILine) {
    const x1 = line.x1;
    const y1 = line.y1;
    const x2 = line.x2;
    const y2 = line.y2;
    const slope = (y2 - y1) / (x2 - x1);
    let a, b, c;
    if (isFinite(slope)) {
        a = -slope;
        b = 1;
    } else {
        a = 1;
        b = 0;
    }
    c = -(a * x1 + b * y1);
    return { a, b, c };
}

function angleInRange(startAngle: number, endAngle: number, angle: number) {
    if (startAngle <= endAngle) {
        return angle >= startAngle && angle <= endAngle;
    }
    return angle >= startAngle || angle <= endAngle;
}

function computeIntersectXY(arc: IArc, line: ILine): Position[] {
    let x1 = line.x1;
    let y1 = line.y1;
    let x2 = line.x2;
    let y2 = line.y2;
    let cx = arc.x;
    let cy = arc.y;
    let r = arc.radius;

    // Move the line segment's start point to the origin and
    // adjust the end point accordingly
    x2 -= x1;
    y2 -= y1;
    cx -= x1;
    cy -= y1;

    // Calculate the coefficients for the quadratic equation
    let a = x2 * x2 + y2 * y2;
    let b = 2 * (x2 * cx + y2 * cy);
    let c = cx * cx + cy * cy - r * r;

    // Calculate the discriminant
    let disc = b * b - 4 * a * c;

    if (disc < 0) {
        return [];
    }

    // Calculate the two solutions for t using the quadratic formula
    // (x1, y1) 是線段的起始點，(x2, y2) 是線段的終點，而 t 是一個在 0 到 1 之間的實數。
    // 當 t = 0 時，我們得到起始點 (x1, y1)，當 t = 1 時，我們得到終點 (x2, y2)，
    // 而當 t 在 0 和 1 之間時，我們得到線段上的某一點。
    let sqrtDisc = Math.sqrt(disc);
    let t1 = (-b - sqrtDisc) / (2 * a);
    let t2 = (-b + sqrtDisc) / (2 * a);

    console.log(`t1=${t1} t2=${t2}`)

    // Points of intersection P1 and P2
    let points = [];

    // Check if the first solution is within the line segment
    if (t1 >= 0 && t1 <= 1) {
        let px = x1 + t1 * x2;
        let py = y1 + t1 * y2;
        let angle = Math.atan2(py - arc.y, px - arc.x);
        if (angle < 0) angle += 2 * Math.PI;  // ensure the angle is between 0 and 2PI

        if (angle >= arc.startAngle && angle <= arc.endAngle) {
            points.push({ x: px, y: py });
        }
    }

    // Check if the second solution is within the line segment
    if (t2 >= 0 && t2 <= 1) {
        let px = x1 + t2 * x2;
        let py = y1 + t2 * y2;
        let angle = Math.atan2(py - arc.y, px - arc.x);
        if (angle < 0) angle += 2 * Math.PI;  // ensure the angle is between 0 and 2PI

        if (angle >= arc.startAngle && angle <= arc.endAngle) {
            points.push({ x: px, y: py });
        }
    }

    return points;
}

class Point {
    pos: Position;
    color: string = "blue";

    constructor(pos: Position) {
        this.pos = pos;
    }
    render(ctx: CanvasRenderingContext2D) {
        const pos = this.pos;
        ctx.fillStyle = this.color;
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, 2 * Math.PI);
        ctx.fill();
    }
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
    center: Point;

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
        this.center = new Point(pos);
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


        const p1 = new Point({ x: pos.x, y: pos.y });
        p1.render(ctx);

        const p2 = new Point(this.getStartAngleXY());
        p2.color = "yellow";
        p2.render(ctx);
        const p3 = new Point(this.getEndAngleXY());
        p3.color = "yellow";
        p3.render(ctx);


        const line = new Line({
            x1: pos.x, y1: pos.y,
            x2: pos.x + roadMargin - roadMargin, y2: pos.y - roadWidth
        });
        line.color = "yellow";
        //line.render(ctx);

        this.center.pos.x += roadLength;
        this.center.pos.y += roadWidth;
        //this.center.render(ctx);
    }

    getStartAngleXY() {
        const pos = this.pos;
        const radius = roadWidth - roadMargin;
        const startAngle = Math.PI;
        return {
            x: pos.x + radius * Math.cos(startAngle),
            y: pos.y + radius * Math.sin(startAngle)
        };
    }

    getEndAngleXY() {
        const pos = this.pos;
        const radius = roadWidth - roadMargin;
        const endAngle = 1.5 * Math.PI;
        return {
            x: pos.x + radius * Math.cos(endAngle),
            y: pos.y + radius * Math.sin(endAngle),
        };
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
    color: string = roadColor;

    constructor(line: ILine) {
        this.line = line;
    }
    render(ctx: CanvasRenderingContext2D) {
        const line = this.line;
        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.lineWidth = 7;
        ctx.strokeStyle = this.color;
        ctx.stroke();
    }
}

class RoadMap {
    pos: Position;
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);

    _line: ILine;

    constructor(pos: Position = { x: 0, y: 0 }) {
        this.pos = pos;
        const curve = new LeftTopCurve()
        this.roads[0][0] = curve;
        this.roads[0][1] = new VerticalRoad();
        this.roads[1][0] = new HorizontalRoad();

        const arc = curve.getArc();
        const line = { x1: 10, y1: 10, x2: 100, y2: 100 };
        const rc = computeIntersectXY(arc, line);
        console.log('no!!', rc);
        this._line = line;
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

        const l = new Line(this._line);
        l.color = "yellow";
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