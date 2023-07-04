import { ILine, IPosition } from "./drawUtils";

export const carWidth = 75;
export const carHeight = 117;
export const carColor = "blue";
export const roadLength = 220;
export const roadWidth = 220;
export const roadColor = 'red';
export const roadMargin = 22;

export class Line {
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


export interface IRoad {
    pos: IPosition;
    render(ctx: CanvasRenderingContext2D, pos: IPosition): void;
}

//垂直
export class VerticalRoad implements IRoad {
    pos: IPosition;

    /**
     * 垂直 
     */
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
export class HorizontalRoad implements IRoad {
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

export class LeftTopCurve implements IRoad {
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


export class RightTopCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x,
            y: pos.y + roadWidth,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, roadWidth - roadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + roadMargin - roadMargin, y, roadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }
}

export class LeftBottomCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x + roadLength,
            y: pos.y,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, roadWidth - roadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + roadMargin - roadMargin, y, roadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }
}

export class RightBottomCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x,
            y: pos.y,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, roadWidth - roadMargin, 0 * Math.PI, 0.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + roadMargin - roadMargin, y, roadMargin, 0 * Math.PI, 0.5 * Math.PI);
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


export class RoadMap {
    pos: IPosition;
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = pos;
        this.roads[0][0] = new LeftTopCurve();
        this.roads[2][0] = new RightTopCurve();
        this.roads[1][0] = new HorizontalRoad();
        this.roads[0][1] = new VerticalRoad();
        this.roads[2][1] = new VerticalRoad();
        this.roads[0][2] = new LeftBottomCurve();
        this.roads[1][2] = new HorizontalRoad();
        this.roads[2][2] = new RightBottomCurve();
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