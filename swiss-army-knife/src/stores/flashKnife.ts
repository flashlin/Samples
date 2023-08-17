import { type IDataTable, fetchAllTable } from './../helpers/tableFetcher';
import { defineStore } from 'pinia';

export interface IPrepareImportDataTable {
  tableName: string;
  dataTable: IDataTable;
}

export interface IFlashKnifeState {
  fullscreenLoading: boolean;
  jsonContent: string;
  dataTableListInWebPage: IPrepareImportDataTable[];
}

const flashKnifeStore = defineStore('flashKnife', {
  state: (): IFlashKnifeState => ({
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

export const useFlashKnifeStore = () => flashKnifeStore();
