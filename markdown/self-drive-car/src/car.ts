import { IPosition, IRect, drawArc, drawRect } from "./drawUtils";
import { Rectangle, CarFrameMargin, CarHeight, CarWidth, CanvasWidth, CanvasHeight, CenterX, CenterY, CarCenterY, CarCenterX, CarPos } from "./gameUtils";
import car1 from './assets/car1.png';
import { Controls } from "./controls";
import { getRectangleWidthHeight, rotateRectangle, updateCoordinates } from "./math";

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
        //ctx.drawImage(this.carImage, x, y, CarWidth, CarHeight);

        const angleInRadians = (this.angle - 270) * (Math.PI / 180);
        // 將原點移至圖像的中心
        ctx.translate(x + CarWidth / 2, y + CarHeight / 2);
        ctx.rotate(angleInRadians);
        ctx.drawImage(this.carImage, -CarWidth / 2, -CarHeight / 2);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    getBound(): IRect {
        let x1 = this.x + CarFrameMargin;
        let y1 = this.y + CarFrameMargin;
        let x2 = x1 + CarWidth - CarFrameMargin;
        let y2 = y1 + CarHeight + CarFrameMargin;
        const [leftTop, rightTop, rightBottom, leftBottom] = rotateRectangle({ x: x1, y: y1 }, { x: x2, y: y2 }, this.angle);
        return { leftTop, rightTop, rightBottom, leftBottom, }
    }

    getFrame(): IRect {
        const [leftTop, rightTop, rightBottom, leftBottom] = rotateRectangle(
            CarPos,
            { x: CarPos.x + CarWidth - CarFrameMargin, y: CarPos.y + CarHeight - CarFrameMargin },
            this.angle);
        return {
            leftTop,
            rightTop,
            rightBottom,
            leftBottom
        };
    }

    move() {
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

        const pos = updateCoordinates({ x: this.x, y: this.y }, this.angle, this.speed);
        this.x = pos.x;
        this.y = pos.y;
    }
}
