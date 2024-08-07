import { drawRect } from "./drawUtils";
import { Rectangle, CarFrameMargin, CarHeight, CarWidth, CarPos, FrameWidth, FrameHeight, CanvasWidth, CanvasHeight, UseBrain, StartX, StartY, RoadMap, EmptyRoad, IRoad, RadarLineLength, objectToArray, ICarState } from "./gameUtils";
import car1 from './assets/car1.png';
import { Controls } from "./controls";
import { ILine, IPosition, IRect, getRectangleWidthHeight, getTwoPointsDistance, rotateRectangle, updateCoordinates } from "./math";
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
    moveDistance = 0;

    x = 0;
    y = 0;

    radar = new Radar();
    brain: Brain;
    prevState = {
        x: 0,
        y: 0,
        angle: 0,
        moveDistance: 0,
    };

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
        this.brain = new Brain(objectToArray(this.getStateObj()));
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

    async move(ctx: CanvasRenderingContext2D, roadMap: RoadMap) {
        const pos0 = { x: this.x, y: this.y };
        this.prevState.x = this.x;
        this.prevState.y = this.y;
        this.prevState.angle = this.angle;
        this.prevState.moveDistance = this.moveDistance;

        const brain = this.brain;
        if (UseBrain) {
            const self = this;
            const action = await brain.control(self.getState.bind(self));
            this.controls.forward = false;
            this.controls.reverse = false;
            this.controls.left = false;
            this.controls.right = false;
            switch (action) {
                case 0:
                    this.controls.forward = true;
                    break;
                case 1:
                    this.controls.left = true;
                    break;
                case 2:
                    this.controls.right = true;
                    break;
                case 3:
                    this.controls.reverse = true;
                    break;
            }
            //console.log('brain', action);
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

        const pos1 = updateCoordinates(pos0, this.angle, this.speed);
        this.x = pos1.x;
        this.y = pos1.y;
        if( this.speed > 0){
            this.moveDistance += this.speed;
        }

        await this.collideAsync(ctx, roadMap);
        this.renderRadarLines(ctx, roadMap);
        return [pos0, pos1];
    }

    renderRadarLines(ctx: CanvasRenderingContext2D, roadMap: RoadMap) {
        const radars = this.radar;
        for (let [index, radarLine] of radars.getBoundLines().entries()) {
            const radar = radars.radarLines[index];
            const [road, collideRadarPoints] = roadMap.collide(ctx, [radarLine]);
            if (collideRadarPoints.length != 0) {
                radar.renderDamaged(ctx, collideRadarPoints[0]);
            } else {
                radar.distance = RadarLineLength;
            }
        }
    }

    getStateObj(): ICarState {
        const radarSense = this.radar.radarLines.map(x => x.distance);
        return {
            x: this.x,
            y: this.y,
            angle: this.angle,
            speed: this.speed,
            damaged: this.damaged ? 1 : 0,
            moveDistance: this.moveDistance,
            radarSense
        }
    }

    getState(): number[] {
        const state = this.getStateObj();
        return objectToArray(state).values;
    }

    async collideAsync(ctx: CanvasRenderingContext2D, roadMap: RoadMap): Promise<[IRoad, IPosition[]]> {
        const boundLines = this.getBoundLines();
        const roads = roadMap.roads;
        for (let ix = 0; ix < roads.length; ix++) {
            for (let iy = 0; iy < roads[ix].length; iy++) {
                const road = roads[ix][iy];
                const collidePoints = road.collide(ctx, boundLines);
                if (collidePoints.length > 0) {
                    this.x = this.prevState.x;
                    this.y = this.prevState.y;
                    this.angle = this.prevState.angle;
                    this.moveDistance = this.prevState.moveDistance;
                    this.speed = 0;
                    this.damaged = true;
                    road.renderDamaged(ctx);

                    if (UseBrain) {
                        await this.brain.control(this.getState.bind(this));
                        this.x = StartX;
                        this.y = StartY;
                        this.angle = 270;
                        this.speed = 0;
                        this.damaged = false;
                        this.moveDistance = 0;
                    }

                    return [road, collidePoints];
                }
            }
        }
        this.damaged = false;
        return [EmptyRoad.Default, []];
    }
}
