<script setup lang="ts">
import Textarea from 'primevue/textarea';
import Button from 'primevue/button';
import Datatable from 'primevue/datatable';
import Column from 'primevue/column';
import Dropdown from 'primevue/dropdown';
import InputText from 'primevue/inputtext';
import Checkbox from 'primevue/checkbox';

import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';

import { reactive, ref } from 'vue';
import '@/models/csv-parser';
import { ClassProperty, CsvStringToInt32, CsvStringToString, VarType, type ICodeConverter, type IDataConverterData } from '@/typings/convert-models';
import type { ICsvReportViewModel } from '@/typings/csv-textarea-models';
import CsvTextArea from '@/components/CsvTextArea.vue';
// defineProps<{
//   msg: string
// }>()

let data = reactive<ICsvReportViewModel>({
  csvTextList: [
      'id,name\n1,flash\n2,jack\n3,Mary',
      `id,addr\n2,Taipei\n3,Kaohsiung`,
    ]
});
</script>

<template>
  <div>
    <h1>CSV Report</h1>
    <CsvTextArea v-for="csvText in data.csvTextList"
      :text="csvText" />
  </div>
</template>

<style scoped>

</style>
