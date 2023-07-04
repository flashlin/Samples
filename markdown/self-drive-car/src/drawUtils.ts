export interface IDrawOptions {
    lineWidth: number,
    strokeSyle: string
}

export const DefaultDrawOptions: IDrawOptions = {
    lineWidth: 7,
    strokeSyle: "red"
}

export function drawArc(ctx: CanvasRenderingContext2D, arc: IArc, 
    options: IDrawOptions=DefaultDrawOptions) {
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