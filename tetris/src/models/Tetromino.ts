export enum MinoType {
    None,
    Solid
}

export abstract class Tetromino
{
    protected _plane: MinoType[][] = [];

    getPlane(): MinoType[][] {
        return this._plane;
    }

    rotate():void {

    }
}

export class StraightPolyomino extends Tetromino
{
    constructor() {
        super();
        this._plane = [
            [ MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid ]
        ];
    }
}