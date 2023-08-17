import { defineStore } from 'pinia';
import { ElLoading } from 'element-plus'
import { fetchAllTable } from '@/helpers/tableFetcher';

interface IJsonKnifeState {
  jsonContent: string;
}

let loadingInstance: any = null;

const jsonKnifeStore = defineStore('jsonKnife', {
  state: (): IJsonKnifeState => ({
    jsonContent: '',
  }),
  getters: {},
  actions: {
    showLoadingFullscreen(toggle: boolean) {
      if (toggle && loadingInstance != null) {
        return;
      }
      if (toggle && loadingInstance == null) {
        loadingInstance = ElLoading.service({ fullscreen: true });
        return;
      }
      loadingInstance.close();
      loadingInstance = null;
    },
    fetchAllDataTable() {
      const allDataTableList = fetchAllTable();
      return allDataTableList;
    }
  },
});

export const useJsonKnifeStore = () => jsonKnifeStore();
