import { defineStore } from 'pinia';

interface IJsonKnifeState {
  jsonContent: string;
}

const jsonKnifeStore = defineStore('jsonKnife', {
  state: (): IJsonKnifeState => ({
    jsonContent: '',
  }),
  getters: {},
  actions: {},
});

export const useJsonKnifeStore = () => jsonKnifeStore();
