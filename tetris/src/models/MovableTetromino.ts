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
