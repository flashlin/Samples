import { drawLine, drawText, lineInfo, posInfo } from "./drawUtils";
import { CarHeight, CarWidth, RadarColor, RadarLine } from "./gameUtils";
import { ILine, IPosition, findTwoLinesIntersection, getDistance, getTwoPointsDistance, rotatePoints } from "./math";

export class Radar {
    pos: IPosition = { x: 0, y: 0 };
    angle = 270;
    center: IPosition = { x: 0, y: 0 };
    carXY: IPosition = { x: 0, y: 0 };

    radarLine: ILine | null = null;

    render(ctx: CanvasRenderingContext2D) {
        const start = {
            x: this.pos.x + CarWidth / 2,
            y: this.pos.y + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLine
        };
        const points = [start, end];
        const [start1, end1] = rotatePoints(this.center, this.angle, points);
        drawLine(ctx, { start: start1, end: end1 }, { strokeSyle: RadarColor, lineWidth: 3 });

        const radarLine = this.getBoundLine()[0];
        drawText(ctx, start1, `${posInfo(radarLine.start)}`)
        drawText(ctx, end1, `${posInfo(radarLine.end)}`)
    }

    getBoundLine() {
        const start = {
            x: this.carXY.x,
            y: this.carXY.y - CarHeight / 2 + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLine
        };
        const [start1, end1] = rotatePoints(this.carXY, this.angle, [start, end]);
        return [{ start: start1, end: end1 }];
    }

    renderDamaged(ctx: CanvasRenderingContext2D, point: IPosition) {
        const { start, end } = this.getBoundLine()[0];
        const distance = getTwoPointsDistance(start, point);
        const startPos = {
            x: this.pos.x + CarWidth / 2,
            y: this.pos.y + 20,
        };
        const endPos = {
            x: startPos.x,
            y: startPos.y - distance,
        };
        const [startPos1, endPos1] = rotatePoints(this.center, this.angle, [startPos, endPos]);
        const radarLine = {
            start: startPos1,
            end: endPos1,
        };
        drawLine(ctx, radarLine, { strokeSyle: 'yellow', lineWidth: 3 });
    }
}