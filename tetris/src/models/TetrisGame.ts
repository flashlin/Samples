import { timer, Observable, Subscription, Subject } from 'rxjs';
import { TetSquareRectangle } from './TetSquareRectangle';
import { MovableTetromino } from './MovableTetromino';
import { StraightPolyomino } from './StraightPolyomino';

export class TetrisGame {
   private _gameTimerSource: Observable<number>;
   _gameTimer: Subscription | undefined;
   gameRect: TetSquareRectangle;

   constructor() {
      this._gameTimerSource = timer(1000, 1000);
      this.gameRect = new TetSquareRectangle(10, 20);
   }

   start(): void {
      this._gameTimer = this._gameTimerSource.subscribe(val => {
         let gameRect = this.gameRect;
         if (gameRect.cube == MovableTetromino.Empty) {
            gameRect.addTetromino(new StraightPolyomino());
         } else {
            if (!gameRect.dropCube()) {
               gameRect.fixCube();
            }
         }
         gameRect.dump();
      });
   }
}
