import { drawLine, posInfo } from "./drawUtils";
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

        if (this.radarLine != null) {
            drawLine(ctx, this.radarLine, { strokeSyle: 'yellow', lineWidth: 3 });
        }
    }

    collide(ctx: CanvasRenderingContext2D, lines: ILine[]) {
        const start = {
            x: this.carXY.x,
            y: this.carXY.y + 20,
        };
        const end = {
            x: start.x,
            y: start.y - RadarLine
        };
        const radarLine = { start, end };
        for (let line of lines) {
            const point = findTwoLinesIntersection(radarLine, line);
            if (point != null) {
                const distance = getTwoPointsDistance(start, point);
                const startPos = {
                    x: this.pos.x + CarWidth / 2,
                    y: this.pos.y + 20,
                };
                this.radarLine = {
                    start: startPos,
                    end: {
                        x: startPos.x,
                        y: startPos.y - distance,
                    }
                };
                drawLine(ctx, this.radarLine, { strokeSyle: 'yellow', lineWidth: 3 });
                return true;
            }
        }
        this.radarLine = null;
        return false;
    }
}