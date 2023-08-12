import { defineStore } from 'pinia';

interface IJsonSnifeState {
  jsonContent: string;
}

export const useCounterStore = defineStore('counter', {
  state: (): IJsonSnifeState => ({
    jsonContent: '',
  }),
  actions: {},
});
