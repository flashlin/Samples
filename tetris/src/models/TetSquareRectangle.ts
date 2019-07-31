import { MinoType, Tetromino } from './Tetromino';
import { MovableTetromino } from './MovableTetromino';

export class TetSquareRectangle {
   dump() {
      let text = '';
      for (let y = 0; y < this.height; y++) {
         for (let x = 0; x < this.height; x++) {
            let box = this._plane[y][x];
            if (box == MinoType.None) {
               text += '.';
            } else {
               text += '*';
            }
         }
         text += '\r\n';
      }
      console.log(text);
   }
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

   dropCube(): boolean {
      let cube = this.cube;
      if (cube == MovableTetromino.Empty) {
         return false;
      }
      if (cube.y + cube.height >= this.height) {
         return false;
      }
      cube.y++;
      if( this.isCollision() ){
         cube.y--;
         return false;
      }
      return true;
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
      this.cube = MovableTetromino.Empty;
   }

   isCollision(): boolean {
      let cube = this.cube;
      let cubeY = 0;
      for (let y = cube.y; y < cube.y + cube.height; y++) {
         let cubeX = 0;
         for (let x = cube.x; x < cube.x + cube.width; x++) {
            let box = cube.getBox(cubeX++, cubeY);
            if (box != MinoType.None) {
               if( this._plane[y][x] != MinoType.None) {
                  return true;
               }
            }
         }
         cubeY++;
      }
      return false;
   }
}
