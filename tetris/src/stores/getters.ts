import { IState } from './state';
import { GetterTree } from 'vuex';

function isEmpty(str: string) {
   return !str || 0 === str.length;
}

const getters: GetterTree<IState, any> = {};

getters.Test = (state: IState): boolean => {
   return true;
};

export default getters;
