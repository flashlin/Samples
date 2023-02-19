import { ref, computed } from "vue";
import { defineStore } from "pinia";

export const useAppState = defineStore("appState", () => {
  const isAuthenticated = ref(false);
  // const doubleCount = computed(() => isLogined.value * 2);
  // function increment() {
  //   isLogined.value++;
  // }

  return { isAuthenticated };
});
