import { Tetromino, MinoType } from './Tetromino';
export class SquarePolyomino extends Tetromino {
   constructor() {
      super();
      this._plane = this.normalizePlane([
         [MinoType.Solid, MinoType.Solid],
         [MinoType.Solid, MinoType.Solid]
      ]);
   }
}

//T-polyomino
//L
//Z
