import { IArc, ILine, IPosition, IRect } from "./math";

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

export function drawLine(ctx: CanvasRenderingContext2D, line: ILine,
    options: Partial<IDrawOptions> = DefaultDrawOptions) {
    const newOptions = {...DefaultDrawOptions, ...options};
    ctx.beginPath();
    ctx.strokeStyle = newOptions.strokeSyle;
    ctx.lineWidth = newOptions.lineWidth;
    ctx.moveTo(line.start.x, line.start.y);
    ctx.lineTo(line.end.x, line.end.y);
    ctx.stroke();
}


export function drawRect(ctx: CanvasRenderingContext2D, rect: IRect,
    options: Partial<IDrawOptions> = DefaultDrawOptions) {
    const newOptions = {...DefaultDrawOptions,...options };
    ctx.beginPath();
    ctx.strokeStyle = newOptions.strokeSyle;
    ctx.lineWidth = newOptions.lineWidth;
    ctx.moveTo(rect.leftTop.x, rect.leftTop.y);
    ctx.lineTo(rect.rightTop.x, rect.rightTop.y);
    ctx.lineTo(rect.rightBottom.x, rect.rightBottom.y);
    ctx.lineTo(rect.leftBottom.x, rect.leftBottom.y);
    ctx.lineTo(rect.leftTop.x, rect.leftTop.y);
    ctx.stroke();
}

export interface IDrawTextOptions {
    font: string;
    fillStyle: string;
    strokeStyle: string;
}

export const DefaultDrawTextOptions: IDrawTextOptions = {
    font: "24px Arial",
    fillStyle: "white",
    strokeStyle: "blue",
};

export function posInfo(pos: IPosition) {
    let x = Math.round(pos.x);
    let y = Math.round(pos.y);
    return `{x:${x},y:${y}}`;
}

export function drawText(ctx: CanvasRenderingContext2D, 
    pos: IPosition, 
    text: string, 
    options: Partial<IDrawTextOptions> = DefaultDrawTextOptions) {
    // 設置文字樣式和大小
    const newOptions = {...DefaultDrawTextOptions,...options };
    ctx.font = newOptions.font;
    ctx.strokeStyle = newOptions.strokeStyle;
    ctx.strokeText(text, pos.x + 1, pos.y + 1);
    ctx.fillStyle = newOptions.fillStyle;
    // 使用fillText方法在(50, 50)位置繪製填充文字
    ctx.fillText(text, pos.x, pos.y);
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

