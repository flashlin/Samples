import { Tetromino, MinoType } from './Tetromino';
export class StraightPolyomino extends Tetromino {
   constructor() {
      super();
      this._plane = [
         [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
      ];
      this._plane = this.normalizePlane(this._plane);
   }
}
