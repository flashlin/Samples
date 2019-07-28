import { MinoType, Tetromino } from "./Tetromino";

export class MovableTetromino {
   getBox(x: number, y: number): MinoType {
      return this.cube.getBox(x, y);
   }
   static Empty: MovableTetromino = new MovableTetromino(Tetromino.Empty);

   constructor(cube: Tetromino) {
      this.cube = cube;
   }

   x: number = 0;
   y: number = 0;
   cube: Tetromino;

   get width(): number {
      return this.cube.width;
   }

   get height(): number {
      return this.cube.height;
   }
}

export class TetSquareRectangle {
   protected _plane: MinoType[][] = [];
   width: number;
   height: number;

   constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
      this.cleanPlane();
   }

   cube: MovableTetromino = MovableTetromino.Empty;
   startX: number = 1;

   getPlane(): MinoType[][] {
      return this._plane;
   }

   dropCube(): void {
      let cube = this.cube;
      if (cube.y + cube.height >= this.height) {
         return;
      }
      cube.y++;
   }

   cleanPlane(): void {
      for (let y = 0; y < this.height; y++) {
         for (let x = 0; x < this.width; x++) {
            this._plane[y] = this._plane[y] || [];
            this._plane[y][x] = MinoType.None;
         }
      }
   }

   addTetromino(mino: Tetromino): void {
      if (this.cube != MovableTetromino.Empty) {
         throw new Error("cube is running, can't add new cube.");
      }

      let newCube = new MovableTetromino(mino);
      newCube.x = this.startX;
      newCube.y = 0;
      this.cube = newCube;
   }

   fixCube(): void {
      let cube = this.cube;
      let cubeY = 0;
      for (let y = cube.y; y < cube.y + cube.height; y++) {
         let cubeX = 0;
         for (let x = cube.x; x < cube.x + cube.width; x++) {
            let box = cube.getBox(cubeX++, cubeY);
            if (box != MinoType.None) {
               this._plane[y][x] = box;
            }
         }
         cubeY++;
      }
   }

   isCollision(): boolean {
      return false;
   }
}
