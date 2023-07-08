import { ILine, IPosition, IRect, drawText, posInfo } from "./drawUtils";

export const CarFrameMargin = 4;
export const CarWidth = 75;
export const CarHeight = 117;
export const CarColor = "blue";
export const RoadLength = 220;
export const RoadWidth = 220;
export const RoadColor = 'blue';
export const RoadMargin = 22;
export const CanvasWidth = 800;
export const CanvasHeight = 700;
export const CenterX = CanvasWidth / 2 - CarWidth / 2;
export const CenterY = CanvasHeight / 2 - CarHeight / 2;
export const CarPos = { x: CenterX + CarFrameMargin, y: CenterY + CarFrameMargin };
export const CarCenterX = CarWidth / 2;
export const CarCenterY = CarHeight / 2;


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
    ix: number;
    iy: number;
    pos: IPosition;
    render(ctx: CanvasRenderingContext2D): void;
    collide(ctx: CanvasRenderingContext2D, rect: IRect): boolean;
    renderDamaged(ctx: CanvasRenderingContext2D): void;
}

//垂直
export class VerticalRoad implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = "";

    /**
     * 垂直 
     */
    constructor() {
    }

    render(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        ctx.beginPath();
        ctx.moveTo(x + RoadMargin, y);
        ctx.lineTo(x + RoadMargin, y + RoadLength);
        ctx.moveTo(x + RoadWidth - RoadMargin, y);
        ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadLength);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;  // 線條顏色
        ctx.stroke();
    }

    renderDamaged(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        if (this.lineDamaged == "line1") {
            ctx.beginPath();
            ctx.moveTo(x + RoadMargin, y);
            ctx.lineTo(x + RoadMargin, y + RoadLength);
            // ctx.moveTo(x + RoadWidth - RoadMargin, y);
            // ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadLength);
            ctx.lineWidth = 7;
            ctx.strokeStyle = "red";
            ctx.stroke();
        }

        if (this.lineDamaged == "line2") {
            ctx.beginPath();
            ctx.moveTo(x + RoadWidth - RoadMargin, y);
            ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadLength);
            ctx.lineWidth = 7;
            ctx.strokeStyle = "red";
            ctx.stroke();
        }
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadLength;
        const line1 = {
            start: {
                x: x + RoadMargin,
                y: y
            },
            end: {
                x: x + RoadMargin,
                y: y + RoadLength
            }
        };

        const points1 = rectangleIntersectLine(rect, line1);
        if (this.ix == 0 && this.iy == 1) {
            drawText(ctx, this.pos, `${posInfo(line1.start)}`)
            drawText(ctx, { x: this.pos.x, y: this.pos.y + RoadLength }, `${posInfo(line1.end)}`)
        }
        if (points1.length > 0) {
            this.lineDamaged = "line1";
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
            }
        };
        const points2 = rectangleIntersectLine(rect, line2);
        if (points2.length > 0) {
            this.lineDamaged = "line2";
            return true;
        }

        this.lineDamaged = "";
        return false;
    }
}


// 水平
export class HorizontalRoad implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };

    render(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        ctx.beginPath();
        ctx.moveTo(x, y + RoadMargin);
        ctx.lineTo(x + RoadLength, y + RoadMargin);
        ctx.moveTo(x, y + RoadWidth - RoadMargin);
        ctx.lineTo(x + RoadLength, y + RoadWidth - RoadMargin);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }
}

export class LeftTopCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };


    render(ctx: CanvasRenderingContext2D) {
        let x = this.pos.x + RoadLength;
        let y = this.pos.y + RoadWidth;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, Math.PI, 1.5 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }
}


export class RightTopCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };

    render(ctx: CanvasRenderingContext2D) {
        let x = this.pos.x;
        let y = this.pos.y + RoadWidth;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 1.5 * Math.PI, 0);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }
}

export class LeftBottomCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };

    render(ctx: CanvasRenderingContext2D) {
        let x = this.pos.x + RoadLength;
        let y = this.pos.y;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 0.5 * Math.PI, 1 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }
}

export class RightBottomCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };

    render(ctx: CanvasRenderingContext2D) {
        let x = this.pos.x;
        let y = this.pos.y;
        ctx.beginPath();
        ctx.arc(x, y, RoadWidth - RoadMargin, 0 * Math.PI, 0.5 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x + RoadMargin - RoadMargin, y, RoadMargin, 0 * Math.PI, 0.5 * Math.PI);
        ctx.strokeStyle = RoadColor;
        ctx.lineWidth = 7;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }
}

export class EmptyRoad implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    render(ctx: CanvasRenderingContext2D) {
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return false;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

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
            const road = roadMap[x][y] = createRoad(ch);
            road.ix = x;
            road.iy = y;
        }
    }
    return roadMap;
}

export class RoadMap {
    pos: IPosition = { x: 0, y: 0 };
    roads: IRoad[][] = create2dArray<IRoad>(10, 10);

    constructor() {
        this.roads = readMap(map1Content);
    }

    render(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                if (road == null) {
                    continue;
                }
                road.pos = {
                    x: x + ix * RoadWidth,
                    y: y + iy * RoadLength,
                };
                road.render(ctx);
            }
        }
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                if (road == null) {
                    continue;
                }
                if (road.collide(ctx, rect)) {
                    return road;
                }
            }
        }
        return null;
    }
}


export class Rectangle {
    pos: IPosition;
    width: number;
    height: number;
    color: string = "blue";

    constructor(pos: IPosition, width: number, height: number) {
        this.pos = pos;
        this.width = width;
        this.height = height;
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        ctx.strokeStyle = this.color;
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