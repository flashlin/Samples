import { defineStore } from 'pinia'

interface IGptState {
  name: string
}

const gptStore = defineStore('gptStore', {
  state: (): IGptState => ({
    name: "",
  }),
  getters: {
    name: state => state.name,
  },
  actions: {
  },
})


export const useGptStore = () => gptStore();