import { ref, computed } from "vue";
import { defineStore } from "pinia";
import type { ILocalQueryClient } from "@/apis/LocalQueryClient";

export const useAppState = defineStore("appState", () => {
  const isAuthenticated = ref(false);
  const guid = ref("");
  const appUid = ref("");
  const appPort = ref(0);
  const localQueryClient = ref<ILocalQueryClient>();
  // const doubleCount = computed(() => isLogined.value * 2);
  // function increment() {
  //   isLogined.value++;
  // }

  return {
    isAuthenticated,
    guid,
    appUid,
    appPort,
    localQueryClient,
  };
});
