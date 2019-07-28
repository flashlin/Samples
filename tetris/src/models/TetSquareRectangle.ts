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
      let y = 0;
      for (let y0 = cube.y; y0 < cube.y + cube.height; y0++) {
			let x = 0;
			for (let x0 = cube.x; x0 < cube.x + cube.width; x0++) {
				let box = cube.getBox(x++, y);
				if( box != MinoType.None ){
					this._plane[y][x] = box;
				}
			}
			y++;
      }
   }

   isCollision(): boolean {
      return false;
   }
}
