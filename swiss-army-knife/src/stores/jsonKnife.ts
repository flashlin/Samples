import { defineStore } from 'pinia';
import { fetchAllTable } from '@/helpers/tableFetcher';

export interface IPrepareImportDataTable {
  tableName: string;
  dataTable: object[];
}

export interface IJsonKnifeState {
  fullscreenLoading: boolean;
  jsonContent: string;
  dataTableListInWebPage: IPrepareImportDataTable[];
}

const jsonKnifeStore = defineStore('jsonKnife', {
  state: (): IJsonKnifeState => ({
    fullscreenLoading: false,
    jsonContent: '',
    dataTableListInWebPage: [],
  }),
  getters: {},
  actions: {
    showLoadingFullscreen(toggle: boolean) {
      this.fullscreenLoading = toggle;
    },
    fetchAllDataTableInWebPage() {
      const allDataTableList = fetchAllTable();
      let index = -1;
      this.dataTableListInWebPage = allDataTableList.map(x => {
        index++;
        return {
          tableName: `table${index}`,
          dataTable: x,
        };
      });
    }
  },
});

export const useJsonKnifeStore = () => jsonKnifeStore();
