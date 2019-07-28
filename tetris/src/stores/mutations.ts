import { MutationTree } from 'vuex';
import types from "./mutation-types";
import { IState } from "./state";

const mutations: MutationTree<any> = {};

mutations[types.LOADING] = (state: IState, msg: string) => {
   state.loadingMessage = msg;
}

export default mutations;