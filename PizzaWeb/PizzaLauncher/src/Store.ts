import { defineStore } from 'pinia';

export const useTodoStore = defineStore({
  id: 'todo',
  state: () => ({ count: 0, title: "Cook noodles", done:false })
});