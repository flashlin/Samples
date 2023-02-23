import { ref, computed } from "vue";
import { defineStore } from "pinia";
import type { ILocalQueryClient } from "@/apis/LocalQueryClient";
import { useLocalQueryClient } from '../apis/LocalQueryClient';

export interface IAppState 
{
  isAuthenticated: boolean;
  guid: string;
  appUid: string;
  appPort: number;
}

export const appStore = defineStore("appStore", {
  state: (): IAppState => ({
    isAuthenticated: false,
    guid: '',
    appUid: '',
    appPort: 0,
  }),
  getters: {
    IsAuthenticated(state) {
      return state.isAuthenticated;
    },
  },
  actions: {
    getLocalQueryClient() {
      const client = useLocalQueryClient();
      client.setConnectOption({
        appUid: this.appUid,
        appPort: this.port,
      });
      return client;
    },
  },
});

export const useAppStore = () => appStore();