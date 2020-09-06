import { Tetromino, MinoType } from './Tetromino';
export class StraightPolyomino extends Tetromino {
   constructor() {
      super();
      this._plane = this.normalizePlane([
         [MinoType.Solid, MinoType.Solid, MinoType.Solid, MinoType.Solid]
      ]);
   }
}
