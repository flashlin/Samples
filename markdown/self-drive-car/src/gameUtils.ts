import { ILine, IPosition, IRect, drawText } from "./drawUtils";

export const CarFrameMargin = 4;
export const CarWidth = 75;
export const CarHeight = 117;
export const CarColor = "blue";
export const RoadLength = 220;
export const RoadWidth = 220;
export const RoadColor = 'red';
export const RoadMargin = 22;
export const CanvasWidth = 800;
export const CanvasHeight = 700;

export class Line {
    line: ILine;
    color: string = RoadColor;

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
        ctx.moveTo(x + RoadMargin, y);
        ctx.lineTo(x + RoadMargin, y + RoadLength);
        ctx.moveTo(x + RoadWidth - RoadMargin, y);
        ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadLength);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;  // 線條顏色
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        const line1 = {
            start : {
            x: x + RoadMargin,
            y: y
        },
        end : {
            x: x + RoadMargin,
            y: y + RoadLength
        }};
        const points1 = rectangleIntersectLine(rect, line1);
        if( points1.length > 0) {
            return true;
        }

        const line2 = {
        start: {
            x: x + RoadWidth - RoadMargin,
            y: y
        },
         end: {
            x: x + RoadWidth - RoadMargin, 
            y: y + RoadLength,
        }};
        const points2 = rectangleIntersectLine(rect, line2);
        if( points2.length > 0 ) {
            return true;
        }
        return false;
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
        ctx.moveTo(x, y + RoadMargin);
        ctx.lineTo(x + RoadLength, y + RoadMargin);
        ctx.moveTo(x, y + RoadWidth - RoadMargin);
        ctx.lineTo(x + RoadLength, y + RoadWidth - RoadMargin);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
    }
}

export class LeftTopCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x + RoadLength,
            y: pos.y + RoadWidth,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
    }
}


export class RightTopCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x,
            y: pos.y + RoadWidth,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
    }
}

export class LeftBottomCurve implements IRoad {
    pos: IPosition;

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = {
            x: pos.x + RoadLength,
            y: pos.y,
        };
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        let x = this.pos.x + pos.x;
        let y = this.pos.y + pos.y;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
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
        ctx.arc(x, y, RoadWidth - RoadMargin, 0 * Math.PI, 0.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 0 * Math.PI, 0.5 * Math.PI);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
    }
}

export class EmptyRoad implements IRoad {
    pos: IPosition = { x: 0, y: 0 };
    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
    }

    collide(pos: IPosition, rect: IRect) {
        return false;
    }
}

function create2dArray<T>(width: number, height: number): T[][] {
    let arr: T[][] = new Array(width);
    for (let i = 0; i < arr.length; i++) {
        arr[i] = new Array(height);
    }
    return arr;
}

import map1Content from '@/assets/map.txt?raw';
import { rectangleIntersectLine } from "./math";

function createRoad(ch: string) {
    const dict: Record<string, () => IRoad> = {
        '-': () => new HorizontalRoad(),
        '|': () => new VerticalRoad(),
        '/': () => new LeftTopCurve(),
        '\\': () => new RightTopCurve(),
        'L': () => new LeftBottomCurve(),
        '+': () => new RightBottomCurve(),
    };
    if (!(ch in dict)) {
        return new EmptyRoad();
    }
    return dict[ch]();
}

function readMap(mapContent: string): IRoad[][] {
    const lines = mapContent.split('\n');
    const width = lines.reduce((max, x) => Math.max(max, x.length), 0);
    const height = lines.length;
    const roadMap: IRoad[][] = create2dArray(width, height);
    for (let y = 0; y < height; y++) {
        let line = lines[y];
        for (let x = 0; x < width; x++) {
            let ch = line[x];
            roadMap[x][y] = createRoad(ch);
        }
    }
    return roadMap;
}

export class RoadMap {
    pos: IPosition;
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);

    constructor(pos: IPosition = { x: 0, y: 0 }) {
        this.pos = pos;
        this.roads = readMap(map1Content);
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
                    x: x + ix * RoadWidth,
                    y: y + iy * RoadLength,
                };
                road.render(ctx, roadPos);
            }
        }
    }
}


export class Rectangle {
    pos: IPosition;
    width: number;
    height: number;

    constructor(pos: IPosition, width: number, height: number) {
        this.pos = pos;
        this.width = width;
        this.height = height;
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 7;
        ctx.strokeRect(x, y, this.width, this.height);
    }
}

export class FPS {
    frameCount = 0;
    fps = 0;
    lastTime = performance.now();

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;

        if (deltaTime >= 1000) { // 更新FPS計數每秒一次
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;
        }

        drawText(ctx, { x: 0, y: 24 }, `FPS: ${this.fps}`);

        this.frameCount++;
    }
}