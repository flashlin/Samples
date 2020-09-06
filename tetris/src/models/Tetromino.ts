export enum MinoType {
   None,
   Solid
}

export class Tetromino {
   getBox(x: number, y: number): MinoType {
      return this._plane[y][x];
   }
   static Empty: Tetromino = new Tetromino();

   protected _plane: MinoType[][] = [];

   get width(): number {
      if (this._plane[0] == undefined) {
         return 0;
      }
      return this._plane[0].length;
   }

   get height(): number {
      return this._plane.length;
   }

   getPlane(): MinoType[][] {
      return this._plane;
   }

   leftRotate(): void {
      let newPlane: MinoType[][] = [];
      let rowLength = this.getRowLength(this._plane);
      let nx = -1;
      for (let y = 0; y < this._plane.length; y++) {
         let ny = rowLength - 1;
         nx++;

         let columns = this._plane[y];
         if (columns === undefined) {
            continue;
         }

         for (let x = 0; x < columns.length; x++) {
            newPlane[ny] = newPlane[ny] || [];
            let old = columns[x];
            if (old !== undefined) {
               newPlane[ny][nx] = columns[x];
            }
            ny--;
         }
      }
      this._plane = this.normalizePlane(newPlane);
   }

   rightRotate(): void {}

   normalizePlane(plane: MinoType[][]): MinoType[][] {
      plane = this.fillPlane(plane);
      plane = this.trimYPlane(plane);
      plane = this.trimXPlane(plane);
      return plane;
   }

   trimYPlane(plane: MinoType[][]): MinoType[][] {
      let rowLength = this.getRowLength(plane);
      let newPlane: MinoType[][] = [];
      for (let y = 0; y < rowLength; y++) {
         let columns = plane[y];
         let countX = 0;
         for (let x = 0; x < rowLength; x++) {
            if (columns[x] === MinoType.None) {
               countX++;
            }
         }
         if (countX !== rowLength) {
            newPlane.push(columns);
         }
      }
      return newPlane;
   }

   trimXPlane(plane: MinoType[][]): MinoType[][] {
      plane = this.leftTrimXPlane(plane);
      plane = this.rightTrimXPlane(plane);
      return plane;
   }

   isNone(data: MinoType): boolean {
      if (data === undefined) {
         return true;
      }
      if (data == MinoType.None) {
         return true;
      }
      return false;
   }

   rightTrimXPlane(plane: MinoType[][]): MinoType[][] {
      let rowLength = this.getRowLength(plane);
      let newPlane: MinoType[][] = [];
      let idx = -1;
      for (let x = rowLength - 1; x >= 0; x--) {
         let countY = 0;
         for (let y = 0; y < rowLength; y++) {
            if (plane[y] === undefined || this.isNone(plane[y][x])) {
               countY++;
            }
         }
         if (countY !== rowLength) {
            idx = x + 1;
            break;
         }
      }

      let ny = 0;
      for (let y = 0; y < rowLength; y++) {
         if (plane[y] !== undefined) {
            newPlane[ny] = plane[y].slice(0, idx);
            ny++;
         }
      }
      return newPlane;
   }

   leftTrimXPlane(plane: MinoType[][]): MinoType[][] {
      let rowLength = this.getRowLength(plane);
      let newPlane: MinoType[][] = [];
      let idx = -1;
      for (let x = 0; x < rowLength; x++) {
         let countY = 0;
         for (let y = 0; y < rowLength; y++) {
            if (plane[y] === undefined || plane[y][x] === MinoType.None) {
               countY++;
            }
         }
         if (countY !== rowLength) {
            idx = x;
            break;
         }
      }
      for (let y = 0; y < rowLength; y++) {
         if (plane[y] !== undefined) {
            newPlane[y] = plane[y].slice(idx);
         }
      }
      return newPlane;
   }

   fillPlane(plane: MinoType[][]): MinoType[][] {
      let rowLength = this.getRowLength(plane);
      let newPlane: MinoType[][] = [];
      for (let y = 0; y < rowLength; y++) {
         let oldColumns = plane[y];
         oldColumns = oldColumns || [];

         newPlane[y] = [];
         let columns = newPlane[y];

         for (let x = 0; x < rowLength; x++) {
            let old = oldColumns[x];
            if (old === undefined) {
               columns[x] = MinoType.None;
            } else {
               columns[x] = old;
            }
         }
      }
      return newPlane;
   }

   getRowLength(plane: MinoType[][]): number {
      let rowLength = plane.length;
      for (let y = 0; y < plane.length; y++) {
         let columns = plane[y];
         if (columns === undefined) {
            continue;
         }
         if (rowLength < columns.length) {
            rowLength = columns.length;
         }
      }
      return rowLength;
   }
}
