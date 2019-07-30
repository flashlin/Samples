import { Dispatch, Commit, ActionTree } from 'vuex';
import { IState } from './state';
import types from './mutation-types';
import { TetrisGame } from '@/models/TetrisGame';

export interface ActionContext {
   dispatch: Dispatch;
   commit: Commit;
   state: IState;
}

export const actions: ActionTree<IState, any> = {};

actions[types.LOADING] = async (context: ActionContext, msg: string) => {
   context.commit(types.LOADING, msg);
};

let game = new TetrisGame();
actions[types.START] = (context: ActionContext) => {
   game.start();
   context.commit(types.START, game);
}

export default actions;
