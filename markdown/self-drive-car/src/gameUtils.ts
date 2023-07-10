import { drawArc, drawLine, drawRect, drawText, lineInfo, posInfo } from "./drawUtils";

export const CarFrameMargin = 5;
export const CarWidth = 75;
export const CarHeight = 117;
export const FrameWidth = CarWidth - CarFrameMargin * 2;
export const FrameHeight = CarHeight - CarFrameMargin * 2;
export const CarColor = "blue";
export const RoadWidth = 220;
export const RoadColor = 'blue';
export const RoadMargin = 22;
export const CanvasWidth = 800;
export const CanvasHeight = 700;
export const CenterX = CanvasWidth / 2 - CarWidth / 2;
export const CenterY = CanvasHeight / 2 - CarHeight / 2;
export const CarPos = { x: CenterX, y: CenterY };
export const StartX = 100;
export const StartY = 400;
export const DamagedColor = "red";
export const RadarLine = 150;
export const RadarCount = 3;
export const RadarColor = 'gray';

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
    collide(ctx: CanvasRenderingContext2D, rect: IRect): IPosition[];
    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null;
    renderDamaged(ctx: CanvasRenderingContext2D): void;
    getBoundLines(): ILine[];
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
        ctx.lineTo(x + RoadMargin, y + RoadWidth);
        ctx.moveTo(x + RoadWidth - RoadMargin, y);
        ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadWidth);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;  // 線條顏色
        ctx.stroke();

        const [line1, line2] = this.getBoundLines();
        drawText(ctx, {x: x+RoadMargin, y: y}, `${posInfo(line1.start)}`);
    }

    renderDamaged(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        ctx.beginPath();
        if (this.lineDamaged == "line1") {
            ctx.moveTo(x + RoadMargin, y);
            ctx.lineTo(x + RoadMargin, y + RoadWidth);
        }
        if (this.lineDamaged == "line2") {
            ctx.moveTo(x + RoadWidth - RoadMargin, y);
            ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadWidth);
        }
        ctx.lineWidth = 7;
        ctx.strokeStyle = "red";
        ctx.stroke();
    }

    getBoundLines() {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadWidth;
        const line1 = {
            start: {
                x: x + RoadMargin,
                y: y
            },
            end: {
                x: x + RoadMargin,
                y: y + RoadWidth
            }
        };

        const line2 = {
            start: {
                x: x + RoadWidth - RoadMargin,
                y: y
            },
            end: {
                x: x + RoadWidth - RoadMargin,
                y: y + RoadWidth,
            }
        };
        return [line1, line2];
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const [line1, line2] = this.getBoundLines();

        const points1 = rectangleIntersectLine(rect, line1);
        // if (this.ix == 0 && this.iy == 1) {
        //     drawText(ctx, this.pos, `${posInfo(line1.start)}`)
        //     drawText(ctx, { x: this.pos.x, y: this.pos.y + RoadWidth }, `${posInfo(line1.end)}`)
        // }
        if (points1.length > 0) {
            this.lineDamaged = "line1";
            return points1;
        }

        const points2 = rectangleIntersectLine(rect, line2);
        if (points2.length > 0) {
            this.lineDamaged = "line2";
            return points2;
        }

        this.lineDamaged = "";
        return [];
    }

    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        const [line1, line2] = this.getBoundLines();
        const point1 = findTwoLinesIntersection(line1, line);
        if( point1 != null ) {
            return point1;
        }
        const point2 = findTwoLinesIntersection(line1, line);
        if( point2 != null ) {
            return point2;
        }
        return null;
    }
}


// 水平
export class HorizontalRoad implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = "";

    render(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        ctx.beginPath();
        ctx.moveTo(x, y + RoadMargin);
        ctx.lineTo(x + RoadWidth, y + RoadMargin);
        ctx.moveTo(x, y + RoadWidth - RoadMargin);
        ctx.lineTo(x + RoadWidth, y + RoadWidth - RoadMargin);
        ctx.lineWidth = 7;
        ctx.strokeStyle = RoadColor;
        ctx.stroke();
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadWidth;
        const line1 = {
            start: {
                x: x,
                y: y + RoadMargin
            },
            end: {
                x: x + RoadWidth,
                y: y + RoadMargin
            }
        };

        const points1 = rectangleIntersectLine(rect, line1);
        if (points1.length > 0) {
            this.lineDamaged = "line1";
            return points1;
        }

        const line2 = {
            start: {
                x: x,
                y: y + RoadWidth - RoadMargin
            },
            end: {
                x: x + RoadWidth,
                y: y + RoadWidth - RoadMargin,
            }
        };
        const points2 = rectangleIntersectLine(rect, line2);
        if (points2.length > 0) {
            this.lineDamaged = "line2";
            return points2;
        }

        this.lineDamaged = "";
        return [];
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {
        const x = this.pos.x;
        const y = this.pos.y;
        ctx.beginPath();
        if (this.lineDamaged == "line1") {
            ctx.moveTo(x, y + RoadMargin);
            ctx.lineTo(x + RoadWidth, y + RoadMargin);
        }
        if (this.lineDamaged == "line2") {
            ctx.moveTo(x, y + RoadWidth - RoadMargin);
            ctx.lineTo(x + RoadWidth, y + RoadWidth - RoadMargin);
        }
        ctx.lineWidth = 7;
        ctx.strokeStyle = "red";
        ctx.stroke();
    }

    getBoundLines() {
        return [];
    }

    
    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }
}

enum CurveType {
    None,
    Outer,
    Inner,
}

const CurveRadius = {
    [CurveType.None]: 0,
    [CurveType.Outer]: RoadWidth - RoadMargin,
    [CurveType.Inner]: RoadMargin,
};

enum CurveRoadType {
    LeftTop,
    RightTop,
    RightBottom,
    LeftBottom
}

/**
 * 圓弧角度
 */
interface CurveAngles {
    startAngle: number;
    endAngle: number;
}

const CurveAngles = {
    [CurveRoadType.LeftTop]: { startAngle: 180, endAngle: 270 },
    [CurveRoadType.RightTop]: { startAngle: 270, endAngle: 360 },
    [CurveRoadType.RightBottom]: { startAngle: 0, endAngle: 90 },
    [CurveRoadType.LeftBottom]: { startAngle: 90, endAngle: 180 },
};

class CurveRoad {
    pos: IPosition = { x: 0, y: 0 };
    type: CurveRoadType;
    angles: CurveAngles;

    constructor(type: CurveRoadType) {
        this.type = type;
        this.angles = CurveAngles[type];
    }

    render(ctx: CanvasRenderingContext2D, color: string): void {
        this.renderCurve(ctx, CurveType.Outer, color);
        this.renderCurve(ctx, CurveType.Inner, color);

        // let x1 = this.pos.x;
        // let y1 = this.pos.y;
        // let x2 = x1 + RoadWidth;
        // let y2 = y1 + RoadWidth;
        // let leftTop = { x: x1, y: y1, };
        // let rightTop = { x: x2, y: y1 };
        // let rightBottom = { x: x2, y: y2 };
        // let leftBottom = { x: x1, y: y2 };
        // drawRect(ctx, { leftTop, rightTop, rightBottom, leftBottom }, { strokeSyle: 'green' });
    }

    renderCurve(ctx: CanvasRenderingContext2D, curveType: CurveType, color: string): void {
        const { x, y } = this.getArcXY();
        if (curveType == CurveType.Outer) {
            const { startAngle, endAngle } = this.angles;
            ctx.beginPath();
            ctx.arc(x, y, CurveRadius[CurveType.Outer], startAngle * Math.PI / 180, endAngle * Math.PI / 180);
            ctx.strokeStyle = color;
            ctx.lineWidth = 7;
            ctx.stroke();
        }

        if (curveType == CurveType.Inner) {
            const { startAngle, endAngle } = this.angles;
            ctx.beginPath();
            ctx.arc(x, y, CurveRadius[CurveType.Inner], startAngle * Math.PI / 180, endAngle * Math.PI / 180);
            ctx.strokeStyle = color;
            ctx.lineWidth = 7;
            ctx.stroke();
        }
    }

    getBoundLines(curveType: CurveType) {
        const arcXY = this.getArcXY();
        const radius = CurveRadius[curveType];
        const { startAngle, endAngle } = this.angles;
        const lines = getArcLines({ pos: arcXY, radius, startAngle, endAngle });
        return lines;
    }

    getAllBoundLines() {
        const lines1 = this.getBoundLines(CurveType.Outer);
        const lines2 = this.getBoundLines(CurveType.Inner);
        return [...lines1, ...lines2];
    }

    collide(rect: IRect) {
        const lines1 = this.getBoundLines(CurveType.Outer);
        for (let line of lines1) {
            //drawLine(ctx, line, { strokeSyle: 'yellow' })
            const points1 = rectangleIntersectLine(rect, line);
            if (points1.length != 0) {
                return { curveType: CurveType.Outer, points: points1 };
            }
        }

        const lines2 = this.getBoundLines(CurveType.Inner);
        for (let line of lines2) {
            const points1 = rectangleIntersectLine(rect, line);
            if (points1.length != 0) {
                return { curveType: CurveType.Inner, points: points1 };
            }
        }

        return { curveType: CurveType.None, points: [] };
    }

    getArcXY() {
        if (this.type == CurveRoadType.LeftTop) {
            return {
                x: this.pos.x + RoadWidth,
                y: this.pos.y + RoadWidth
            }
        }

        if (this.type == CurveRoadType.RightTop) {
            return {
                x: this.pos.x,
                y: this.pos.y + RoadWidth
            }
        }

        if (this.type == CurveRoadType.RightBottom) {
            return {
                x: this.pos.x,
                y: this.pos.y
            }
        }

        return {
            x: this.pos.x + RoadWidth,
            y: this.pos.y
        }
    }
}

/**
 * 左上角圓弧
 */
export class LeftTopCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = CurveType.None;
    curve = new CurveRoad(CurveRoadType.LeftTop);

    render(ctx: CanvasRenderingContext2D) {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.render(ctx, RoadColor);
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        const { curveType, points } = curve.collide(rect);
        this.lineDamaged = curveType;
        return points;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.renderCurve(ctx, this.lineDamaged, DamagedColor);
    }

    getBoundLines() {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        return curve.getAllBoundLines();
    }

    getBoundPos() {
        return {
            x: this.ix * RoadWidth,
            y: this.iy * RoadWidth,
        };
    }
    
    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }
}

/**
 * 右上角圓弧
 */
export class RightTopCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = CurveType.None;
    curve = new CurveRoad(CurveRoadType.RightTop);

    render(ctx: CanvasRenderingContext2D) {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.render(ctx, RoadColor);
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.renderCurve(ctx, this.lineDamaged, DamagedColor);
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        const { curveType, points } = curve.collide(rect);
        this.lineDamaged = curveType;
        return points;
    }
    
    
    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }

    getBoundLines() {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        return curve.getAllBoundLines();
    }

    getBoundPos() {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadWidth;
        return { x, y };
    }
}

/**
 * 左下角圓弧
 */
export class LeftBottomCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = CurveType.None;
    curve = new CurveRoad(CurveRoadType.LeftBottom);

    render(ctx: CanvasRenderingContext2D) {
        const curveRoad = this.curve;
        curveRoad.pos = this.pos;
        curveRoad.render(ctx, RoadColor);
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {
        const curveRoad = this.curve;
        curveRoad.pos = this.pos;
        if (this.lineDamaged == CurveType.Outer) {
            curveRoad.renderCurve(ctx, CurveType.Outer, DamagedColor);
        }
        if (this.lineDamaged == CurveType.Inner) {
            curveRoad.renderCurve(ctx, CurveType.Inner, DamagedColor);
        }
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        const { curveType, points } = curve.collide(rect);
        this.lineDamaged = curveType;
        return points;
    }
    
    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }

    getBoundLines() {
        this.curve.pos = this.getBoundPos();
        return this.curve.getAllBoundLines();
    }

    getBoundPos() {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadWidth;
        return { x, y };
    }
}

/**
 * 右下角圓弧
 */
export class RightBottomCurve implements IRoad {
    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    lineDamaged = CurveType.None;
    curve = new CurveRoad(CurveRoadType.RightBottom);

    render(ctx: CanvasRenderingContext2D) {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.render(ctx, RoadColor);
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {
        const curve = this.curve;
        curve.pos = this.pos;
        curve.renderCurve(ctx, this.lineDamaged, DamagedColor);
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        const { curveType, points } = curve.collide(rect);
        this.lineDamaged = curveType;
        return points;
    }

    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }

    getBoundLines() {
        const curve = this.curve;
        curve.pos = this.getBoundPos();
        return curve.getAllBoundLines();
    }

    getBoundPos() {
        const x = this.ix * RoadWidth;
        const y = this.iy * RoadWidth;
        return { x, y };
    }
}

export class EmptyRoad implements IRoad {
    static Default = new EmptyRoad();

    ix = 0;
    iy = 0;
    pos: IPosition = { x: 0, y: 0 };
    render(ctx: CanvasRenderingContext2D) {
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect) {
        return [];
    }

    collideLine(ctx: CanvasRenderingContext2D, line: ILine): IPosition | null {
        return null;
    }

    renderDamaged(ctx: CanvasRenderingContext2D): void {

    }

    getBoundLines() {
        return [];
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
import { ILine, IPosition, IRect, findTwoLinesIntersection, getArcLines, rectangleIntersectLine } from "./math";

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
                    y: y + iy * RoadWidth,
                };
                road.render(ctx);
            }
        }
    }

    collide(ctx: CanvasRenderingContext2D, rect: IRect): [IRoad, IPosition[]] {
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                const collidePoints = road.collide(ctx, rect);
                if (collidePoints.length > 0) {
                    return [road, collidePoints];
                }
            }
        }
        return [EmptyRoad.Default, []];
    }
    
    collideRadarLine(ctx: CanvasRenderingContext2D, radarLine: ILine): [IRoad, IPosition|null] {
        const roads = this.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                const collidePoints = road.collideLine(ctx, radarLine);
                if (collidePoints != null) {
                    return [road, collidePoints];
                }
            }
        }
        return [EmptyRoad.Default, null];
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