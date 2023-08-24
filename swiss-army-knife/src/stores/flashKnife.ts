import type { IDataTable, MessageType } from '@/helpers/dataTypes';
import { fetchAllTable } from './../helpers/tableFetcher';
import { defineStore } from 'pinia';
import { ElNotification } from 'element-plus';

export interface IPrepareImportDataTable {
  tableName: string;
  dataTable: IDataTable;
}

export interface IFlashKnifeState {
  fullscreenLoading: boolean;
  jsonContent: string;
  tableListInWebPage: IPrepareImportDataTable[];
}

const flashKnifeStore = defineStore('flashKnife', {
  state: (): IFlashKnifeState => ({
    fullscreenLoading: false,
    jsonContent: '',
    tableListInWebPage: [],
  }),
  getters: {},
  actions: {
    showLoadingFullscreen(toggle: boolean) {
      this.fullscreenLoading = toggle;
    },
    fetchAllDataTableInWebPage() {
      const allDataTableList = fetchAllTable();
      let index = -1;
      this.tableListInWebPage = allDataTableList.map(x => {
        index++;
        return {
          tableName: `tb${index}`,
          dataTable: x,
        };
      });
    },
    notify(messageType: MessageType, message: string) {
      ElNotification({
        title: 'Success',
        message: message,
        type: messageType,
        position: 'top-right',
      });
    },
  },
});

export const useFlashKnifeStore = () => flashKnifeStore();
