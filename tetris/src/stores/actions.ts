import { Dispatch, Commit, ActionTree } from 'vuex';
import { IState } from './state';
import types from './mutation-types';

export interface ActionContext {
   dispatch: Dispatch;
   commit: Commit;
   state: IState;
}

export const actions: ActionTree<IState, any> = {};

actions[types.LOADING] = async (context: ActionContext, msg: string) => {
   context.commit(types.LOADING, msg);
};

export default actions;
