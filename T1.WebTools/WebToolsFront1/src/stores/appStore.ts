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
    IsAuthenticated(state) {
      return state.isAuthenticated;
    },
  },
  actions: {
    setLocalQueryClient(client: ILocalQueryClient) {
      this.$patch({ localQueryClient: client });
    },
    getLocalQueryClient() {
      return this.localQueryClient;
    },
  },
});

export const useAppStore = () => appStore();