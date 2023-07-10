import { drawLine, drawText, lineInfo, posInfo } from "./drawUtils";
import { CarHeight, CarWidth, RadarColor, RadarLineLength } from "./gameUtils";
import { EmptyPosition, ILine, IPosition, findTwoLinesIntersection, getDistance, getLineSlope, getTwoPointsDistance, rotatePoints } from "./math";

export class RadarLine {
    pos: IPosition = EmptyPosition;
    carXY: IPosition = EmptyPosition;
    carAngle: number = 0;
    angle: number = 0;

    render(ctx: CanvasRenderingContext2D) {
        const { start, end } = this.getDrawLine();
        const [start1, end1] = rotatePoints(this.pos, this.carAngle + this.angle, [start, end]);
        drawLine(ctx, { start: start1, end: end1 }, { strokeSyle: RadarColor, lineWidth: 3 });

        //draw dump info
        // const radarLine = this.getBoundLine();
        // drawText(ctx, start1, `${posInfo(radarLine.start)}`)
        // drawText(ctx, end1, `${posInfo(radarLine.end)}`)
    }

    renderDamaged(ctx: CanvasRenderingContext2D, point: IPosition) {
        const { start, end } = this.getBoundLine();
        const distance = getTwoPointsDistance(start, point);
        const { start: startDrawPos, end: endDrawPos } = this.getDrawLine();
        const endPos = {
            x: startDrawPos.x,
            y: startDrawPos.y - distance,
        };
        const [startPos1, endPos1] = rotatePoints(this.pos, this.carAngle + this.angle, [startDrawPos, endPos]);
        const radarLine = {
            start: startPos1,
            end: endPos1,
        };
        drawLine(ctx, radarLine, { strokeSyle: 'yellow', lineWidth: 3 });
    }

    getDrawLine() {
        const start = {
            x: this.pos.x,
            y: this.pos.y - CarHeight / 2 + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLineLength
        };
        return { start, end };
    }

    getBoundLine() {
        const start = {
            x: this.carXY.x,
            y: this.carXY.y - CarHeight / 2 + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLineLength
        };
        const [start1, end1] = rotatePoints(this.carXY, this.carAngle + this.angle, [start, end]);
        return { start: start1, end: end1 };
    }

    isMy(point: IPosition): boolean {
        const { start, end } = this.getBoundLine();
        const slope = getLineSlope({ start, end });
        const testLine = { start, end: point };
        const testSlope = getLineSlope(testLine);
        if (this.compareWithTolerance(slope, testSlope)) {
            return true;
        }
        return false;
    }

    compareWithTolerance(a: number, b: number) {
        return Math.abs(a - b) < 0.0000001;
    }
}

function generateAngles(count: number, angle: number): number[] {
    const angles = [];
    let startAngle = 0;

    let counter = 0;
    if (count % 2 != 0) {
        angles.push(startAngle);
        counter = 1;
    }

    for (let i = 1; i < count; i++) {
        if (i % 2 === 1) {
            angles.push(startAngle - counter * angle);  // 遞減角度
        } else {
            angles.push(startAngle + counter * angle);  // 遞增角度
            counter++;
        }
    }

    return angles;
}

export class Radar {
    pos: IPosition = { x: 0, y: 0 };
    carAngle = 270;
    carXY: IPosition = { x: 0, y: 0 };

    radarLines: RadarLine[] = [];

    constructor() {
        const radarLineCount = 5;
        for(let radarAngle of generateAngles(radarLineCount, 15)) {
            const radarLine = new RadarLine();
            radarLine.angle = radarAngle;
            this.radarLines.push(radarLine);
        }
    }

    render(ctx: CanvasRenderingContext2D) {
        for (let radarLine of this.radarLines) {
            radarLine.pos = this.pos;
            radarLine.carXY = this.carXY
            radarLine.carAngle = this.carAngle;
            radarLine.render(ctx);
        }
    }

    getBoundLines(): ILine[] {
        const boundLines: ILine[] = [];
        for (let radarLine of this.radarLines) {
            radarLine.pos = this.pos;
            radarLine.carXY = this.carXY
            radarLine.carAngle = this.carAngle;
            boundLines.push(radarLine.getBoundLine());
        }
        return boundLines;
    }

    renderDamaged(ctx: CanvasRenderingContext2D, point: IPosition) {
        for (let radarLine of this.radarLines) {
            radarLine.pos = this.pos;
            radarLine.carXY = this.carXY
            radarLine.carAngle = this.carAngle;
            if (radarLine.isMy(point)) {
                radarLine.renderDamaged(ctx, point);
            }
        }
    }
}