import { defineStore } from 'pinia';
import { DataTable as DataTableType } from '@/tools/dataTypes';

export const useSupportStore = defineStore('support', {
  state: () => ({
    queryResult: null as DataTableType | null,
  }),
  actions: {
    setQueryResult(result: DataTableType | null) {
      this.queryResult = result;
    },
  },
}); 