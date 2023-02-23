import { ref, computed } from "vue";
import { defineStore } from "pinia";
import type { ILocalQueryClient } from "@/apis/LocalQueryClient";
import { LocalQueryClient } from '../apis/LocalQueryClient';

export interface IAppState 
{
  isAuthenticated: boolean;
  guid: string;
  appUid: string;
  appPort: number;
  localQueryClient: ILocalQueryClient | null;
}


export const appStore = defineStore("appStore", {
  state: (): IAppState => ({
    isAuthenticated: false,
    guid: '',
    appUid: '',
    appPort: 0,
    localQueryClient: null,
  }),
  getters: {
    getIsAuthenticated(state) {
      return state.isAuthenticated;
    },
    getLocalQueryClient(state) {
      return state.localQueryClient;
    },
  },
  actions: {},
});

export const useAppStore = () => appStore();