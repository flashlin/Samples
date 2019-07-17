export enum MinoType {
   None,
   Solid,
   T,
   F
}

export abstract class Tetromino {
   protected _plane: MinoType[][] = [];

   getPlane(): MinoType[][] {
      return this._plane;
   }

   leftRotate(): void {
      let newPlane: MinoType[][] = [];
      let rowLength = this.getRowLength();
      for (let y = 0; y < this._plane.length; y++) {
         let nx = 0;
         let ny = rowLength - 1;
         for (let x = 0; x < this._plane[y].length; x++) {
            newPlane[ny] = [];
            let old = this._plane[y][x];
            if( old !== undefined) {
                newPlane[ny][nx] = this._plane[y][x];
            }
            nx++;
         }
         ny--;
      }
      this._plane = newPlane;
   }

   rightRotate(): void {}

   getRowLength(): number {
      let rowLength = this._plane.length;
      for (let y = 0; y < this._plane.length; y++) {
         if (rowLength < this._plane[y].length) {
            rowLength = this._plane[y].length;
         }
      }
      return rowLength;
   }
}

export class StraightPolyomino extends Tetromino {
   constructor() {
      super();
      this._plane = [
         [MinoType.Solid, MinoType.Solid, MinoType.T, MinoType.F]
      ];
   }
}
