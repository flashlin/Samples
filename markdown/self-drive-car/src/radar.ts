import { drawLine, drawText, lineInfo, posInfo } from "./drawUtils";
import { CarHeight, CarWidth, RadarColor, RadarLineLength } from "./gameUtils";
import { EmptyPosition, ILine, IPosition, findTwoLinesIntersection, getDistance, getLineSlope, getTwoPointsDistance, rotatePoints } from "./math";

export class RadarLine {
    pos: IPosition = EmptyPosition;
    carXY: IPosition = EmptyPosition;
    angle: number = 0;

    render(ctx: CanvasRenderingContext2D) {
        const { start, end } = this.getDrawLine();
        const [start1, end1] = rotatePoints(this.pos, this.angle, [start, end]);
        drawLine(ctx, { start: start1, end: end1 }, { strokeSyle: RadarColor, lineWidth: 3 });

        //draw dump info
        //const radarLine = this.getBoundLine();
        //drawText(ctx, start1, `${posInfo(radarLine.start)}`)
        //drawText(ctx, end1, `${posInfo(radarLine.end)}`)
    }

    renderDamaged(ctx: CanvasRenderingContext2D, point: IPosition) {
        const { start, end } = this.getBoundLine();
        const distance = getTwoPointsDistance(start, point);
        const { start: startDrawPos, end: endDrawPos } = this.getDrawLine();
        const endPos = {
            x: startDrawPos.x,
            y: startDrawPos.y - distance,
        };
        const [startPos1, endPos1] = rotatePoints(this.pos, this.angle, [startDrawPos, endPos]);
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
        const [start1, end1] = rotatePoints(this.carXY, this.angle, [start, end]);
        return { start: start1, end: end1 };
    }
    
    isMy(point: IPosition): boolean {
        const { start, end } = this.getBoundLine();
        const slope = getLineSlope({ start, end });
        const testLine = { start, end: point };
        const testSlope = getLineSlope(testLine);
        if( slope === testSlope ) {
            return true;
        }
        return false;
    }
}

export class Radar {
    pos: IPosition = { x: 0, y: 0 };
    angle = 270;
    carXY: IPosition = { x: 0, y: 0 };

    radarLine: ILine | null = null;

    render(ctx: CanvasRenderingContext2D) {
        const { start, end } = this.getDrawLine();
        const [start1, end1] = rotatePoints(this.pos, this.angle, [start, end]);
        drawLine(ctx, { start: start1, end: end1 }, { strokeSyle: RadarColor, lineWidth: 3 });

        const radarLine = this.getBoundLines()[0];
        drawText(ctx, start1, `${posInfo(radarLine.start)}`)
        drawText(ctx, end1, `${posInfo(radarLine.end)}`)
    }

    getBoundLines() {
        const start = {
            x: this.carXY.x,
            y: this.carXY.y - CarHeight / 2 + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLineLength
        };
        const [start1, end1] = rotatePoints(this.carXY, this.angle, [start, end]);
        return [{ start: start1, end: end1 }];
    }

    renderDamaged(ctx: CanvasRenderingContext2D, point: IPosition) {
        const { start, end } = this.getBoundLines()[0];
        const distance = getTwoPointsDistance(start, point);
        const { start: startDrawPos, end: endDrawPos } = this.getDrawLine();
        const endPos = {
            x: startDrawPos.x,
            y: startDrawPos.y - distance,
        };
        const [startPos1, endPos1] = rotatePoints(this.pos, this.angle, [startDrawPos, endPos]);
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
}