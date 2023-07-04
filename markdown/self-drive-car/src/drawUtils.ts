export type IPosition = {
    x: number,
    y: number,
};

export type IArc = {
    pos: IPosition,
    radius: number,
    startAngle: number,
    endAngle: number
};

export type ILine = {
    start: IPosition,
    end: IPosition,
};

export interface IDrawOptions {
    lineWidth: number,
    strokeSyle: string
}

export const DefaultDrawOptions: IDrawOptions = {
    lineWidth: 7,
    strokeSyle: "red"
}

export function isSamePoint(p1: IPosition, p2: IPosition): boolean {
    if (p1.x == p2.x && p1.y == p2.y) {
        return true;
    }
    return false;
}

export function drawArc(ctx: CanvasRenderingContext2D, arc: IArc,
    options: IDrawOptions = DefaultDrawOptions) {
    ctx.beginPath();
    ctx.strokeStyle = options.strokeSyle;
    ctx.lineWidth = options.lineWidth;
    for (let angle = arc.startAngle; angle <= arc.endAngle; angle += 1) {
        let randi = angle * (Math.PI / 180);
        let x = arc.pos.x + arc.radius * Math.cos(randi);
        let y = arc.pos.y + arc.radius * Math.sin(randi);
        if (angle == arc.startAngle) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}


export class Point {
    pos: IPosition;
    color: string = "blue";

    constructor(pos: IPosition) {
        this.pos = pos;
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x;
        const y = this.pos.y + pos.y;
        ctx.fillStyle = this.color;
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.arc(x, y, ctx.lineWidth / 2, 0, 2 * Math.PI);
        ctx.fill();
    }
}