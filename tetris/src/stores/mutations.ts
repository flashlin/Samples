import { MutationTree } from 'vuex';
import types from './mutation-types';
import { IState } from './state';
import { TetrisGame } from '@/models/TetrisGame';

const mutations: MutationTree<any> = {};

mutations[types.LOADING] = (state: IState, msg: string) => {
   state.loadingMessage = msg;
};

mutations[types.START] = (state: IState, game: TetrisGame) =>{
   state.gameRect = game.gameRect.getPlane();
}

export default mutations;
