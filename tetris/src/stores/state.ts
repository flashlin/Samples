import { MinoType } from '@/models/Tetromino';
import { TetrisGame } from '@/models/TetrisGame';

export interface IState {
   loadingMessage: string;
   gameRect: MinoType[][];
}

const state: IState = {
   loadingMessage: '',
   gameRect: []
};

let game = new TetrisGame();
state.gameRect = game.gameRect.getPlane();

export default state;
