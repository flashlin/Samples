import { drawRect } from "./drawUtils";
import { Rectangle, CarFrameMargin, CarHeight, CarWidth, CarPos, FrameWidth, FrameHeight, CanvasWidth, CanvasHeight, UseBrain } from "./gameUtils";
import car1 from './assets/car1.png';
import { Controls } from "./controls";
import { ILine, IPosition, IRect, getRectangleWidthHeight, rotateRectangle, updateCoordinates } from "./math";
import { Radar } from "./radar";
import { Brain } from "./brain";

export class Car {
    carImage: HTMLImageElement;
    pos: IPosition = { x: 0, y: 0 };
    frame: Rectangle;

    controls: Controls;
    speed = 0;
    acceleration = 0.3;
    maxSpeed = 4;
    friction = 0.02;
    angle = 270;
    damaged = false;

    x = 0;
    y = 0;

    radar = new Radar();
    brain = new Brain();

    constructor() {
        this.carImage = new Image();
        this.carImage.src = car1;
        this.frame = new Rectangle(
            {
                x: this.pos.x + CarFrameMargin,
                y: this.pos.y + CarFrameMargin
            },
            CarWidth - CarFrameMargin * 2,
            CarHeight - CarFrameMargin * 2);
        this.controls = new Controls();
    }

    render(ctx: CanvasRenderingContext2D) {
        const x = this.pos.x;
        const y = this.pos.y;
        //ctx.globalCompositeOperation = 'destination-atop';
        const angleInRadians = (this.angle - 270) * (Math.PI / 180);
        // 將原點移至圖像的中心
        ctx.translate(x + CarWidth / 2, y + CarHeight / 2);
        ctx.rotate(angleInRadians);
        ctx.drawImage(this.carImage, -CarWidth / 2, -CarHeight / 2);
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        //console.log(`car ${this.x} ${this.y}`);
        const radar = this.radar;
        radar.carXY = { x: this.x, y: this.y };
        radar.carAngle = this.angle;
        radar.pos = {
            x: this.pos.x + CarWidth / 2,
            y: this.pos.y + CarHeight / 2,
        }
        radar.render(ctx);

        // 印出 sensor 數字
        //const sensors = [];
        //for(let radarLine of radar.radarLines){
        //    sensors.push(radarLine.distance);
        //}
        //console.log(`sensors ${sensors.join(', ')}`)
    }

    getBoundLines(): ILine[] {
        const { leftTop, rightTop, rightBottom, leftBottom } = this.getBound();
        return [
            { start: leftTop, end: rightTop },
            { start: rightTop, end: rightBottom },
            { start: rightBottom, end: leftBottom },
            { start: leftBottom, end: leftTop },
        ];
    }

    getBound(): IRect {
        let x1 = this.x - FrameWidth / 2;
        let y1 = this.y - FrameHeight / 2;
        let x2 = x1 + FrameWidth;
        let y2 = y1 + FrameHeight;
        const [leftTop, rightTop, rightBottom, leftBottom] = rotateRectangle({ x: x1, y: y1 }, { x: x2, y: y2 }, this.angle);
        return { leftTop, rightTop, rightBottom, leftBottom, }
    }

    getFrame(): IRect {
        let x1 = CanvasWidth / 2 - FrameWidth / 2;
        let y1 = CanvasHeight / 2 - FrameHeight / 2;
        let x2 = x1 + FrameWidth;
        let y2 = y1 + FrameHeight;
        const [leftTop, rightTop, rightBottom, leftBottom] = rotateRectangle(
            { x: x1, y: y1 },
            { x: x2, y: y2 },
            this.angle);
        return {
            leftTop,
            rightTop,
            rightBottom,
            leftBottom
        };
    }

    getDrawBound(): IRect {
        let x1 = CanvasWidth / 2 - FrameWidth / 2;
        let y1 = CanvasHeight / 2 - FrameHeight / 2;
        let x2 = x1 + FrameWidth;
        let y2 = y1 + FrameHeight;
        const [leftTop, rightTop, rightBottom, leftBottom] = rotateRectangle({ x: x1, y: y1 }, { x: x2, y: y2 }, this.angle);
        return { leftTop, rightTop, rightBottom, leftBottom, }
    }

    drawFrame(ctx: CanvasRenderingContext2D) {
        drawRect(ctx, this.getFrame(), { lineWidth: 5, strokeSyle: "yellow" });
        //const carBound = this.getBound();
        // const { leftTop: p0, rightTop: p1, rightBottom: p2, leftBottom: p3 } = this.getDrawBound();
        // drawText(ctx, p0, `${posInfo(carBound.leftTop)}`)
        // drawText(ctx, p1, `${posInfo(carBound.rightTop)}`)
        // drawText(ctx, p2, `${posInfo(carBound.leftBottom)}`)
        // drawText(ctx, p3, `${posInfo(carBound.rightBottom)}`)
    }

    async move() {
        const brain = this.brain;
        if (UseBrain) {
            const action = await brain.control(() => {
                const distances = this.radar.radarLines.map(x => x.distance);
                return [this.damaged ? 1 : 0, this.speed, ...distances];
            });
            console.log('brain', action);
        }

        if (this.controls.forward) {
            this.speed += this.acceleration;
        }

        if (this.controls.reverse) {
            this.speed -= this.acceleration;
        }

        if (this.speed > this.maxSpeed) {
            this.speed = this.maxSpeed;
        }

        if (this.speed < -this.maxSpeed / 2) {
            this.speed = -this.maxSpeed / 2;
        }

        //自動減速效果
        if (this.speed > 0) {
            this.speed -= this.friction;
        }
        if (this.speed < 0) {
            this.speed += this.friction;
        }
        if (Math.abs(this.speed) < this.friction) {
            this.speed = 0;
        }

        if (this.controls.left) {
            this.angle -= 1;
        }

        if (this.controls.right) {
            this.angle += 1;
        }

        const pos0 = { x: this.x, y: this.y };
        const pos1 = updateCoordinates(pos0, this.angle, this.speed);
        this.x = pos1.x;
        this.y = pos1.y;
        return [pos0, pos1];
    }
}
