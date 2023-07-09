import { drawLine } from "./drawUtils";
import { CarWidth, RadarColor, RadarLine } from "./gameUtils";
import { IPosition, rotatePoints } from "./math";

export class Radar {
    pos: IPosition = { x: 0, y: 0 };
    angle = 270;
    center: IPosition = { x: 0, y: 0 };

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
    }
}