export enum MinoType {
   None,
   Solid
}

export abstract class Tetromino {
   protected _plane: MinoType[][] = [];

   getPlane(): MinoType[][] {
      return this._plane;
   }

   leftRotate(): void {
      let newPlane: MinoType[][] = [];
      let rowLength = this.getRowLength();
      let nx = -1;
      for (let y = 0; y < this._plane.length; y++) {
         let ny = rowLength - 1;
         nx++;

         let columns = this._plane[y];
         if( columns === undefined) {
            continue;
         }

         for (let x = 0; x < columns.length; x++) {
            newPlane[ny] = newPlane[ny] || [];
            let old = columns[x];
            if( old !== undefined) {
               newPlane[ny][nx] = columns[x];
            }
            ny--;
         }
      }
      this._plane = newPlane;
   }

   rightRotate(): void {}

   getRowLength(): number {
      let rowLength = this._plane.length;
      for (let y = 0; y < this._plane.length; y++) {
         let columns = this._plane[y];
         if( columns === undefined) {
            continue;
         } 
         if (rowLength < columns.length) {
            rowLength = columns.length;
         }
      }
      return rowLength;
   }
}
