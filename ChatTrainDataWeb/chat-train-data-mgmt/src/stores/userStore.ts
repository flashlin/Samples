import { defineStore } from 'pinia'

interface ITrainData {
  id: number
  instruction: string
  input: string
  output: string
}

interface IUserState {
  name: string
}

const userStore = defineStore('counter', {
  state: (): IUserState => ({
    name: "",
  }),
  getters: {
    name: state => state.name,
  },
  actions: {
  },
})

export const useUserStore = () => userStore()