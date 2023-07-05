import { IPosition } from "./drawUtils";
import { Rectangle, CarFrameMargin, CarHeight, CarWidth, CanvasWidth, CanvasHeight } from "./gameUtils";
import car1 from './assets/car1.png';
import { Controls } from "./controls";
import { updateCoordinates } from "./math";

export class Car {
    carImage: HTMLImageElement;
    pos: IPosition;
    frame: Rectangle;

    controls: Controls;
    speed = 0;
    acceleration = 0.2;
    maxSpeed = 3;
    friction = 0.03;
    angle = 0;
    damaged = false;

    x = 0;
    y = 0;

    constructor() {
        this.pos = {
            x: 0,
            y: 0
        };
        this.carImage = new Image();
        this.carImage.src = car1;
        this.frame = new Rectangle({
            x: this.pos.x + CarFrameMargin,
            y: this.pos.y + CarFrameMargin
        },
            CarWidth - CarFrameMargin * 2,
            CarHeight - CarFrameMargin * 2);
        this.controls = new Controls();
    }

    render(ctx: CanvasRenderingContext2D, pos: IPosition) {
        const x = this.pos.x + pos.x + this.x;
        const y = this.pos.y + pos.y + this.y;
        //ctx.globalCompositeOperation = 'destination-atop';
        //ctx.drawImage(this.carImage, x, y, CarWidth, CarHeight);

        const angleInRadians = this.angle * (Math.PI / 180);
        // 將原點移至圖像的中心
        ctx.translate(x + CarWidth / 2, y + CarHeight / 2);
        ctx.rotate(angleInRadians);
        this.frame.render(ctx, { x: -CarWidth / 2, y: -CarHeight / 2 });
        ctx.drawImage(this.carImage, -CarWidth / 2, -CarHeight / 2);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
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

        const p = updateCoordinates(this.pos, this.angle, this.speed);
        this.pos.x = p.x;
        this.pos.y = p.y;
    }
}