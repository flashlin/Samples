<script setup lang="ts">
import CodeExitor from "@/components/CodeEditor.vue";
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
      {
        name: 'customer',
        text:'id,name\n1,flash\n2,jack\n3,Mary',
        json: {},
      },
      {
        name: 'home',
        text: `id,addr\n2,Taipei\n3,Kaohsiung`,
        json: {},
      },
    ],
  code: `function find(id) {
    return json.home.find(x => x.id == id) != null;
  }
  json.customer.forEach( item => {
    item.flag = find(item.id);
  });
  alert(JSON.stringify(json.customer));`,
});

function evil(json: any, code: string) {
  return new Function('json', code)(json);
}

const onRun = () => {
  let json: any = {};
  data.csvTextList.forEach(csv => {
    csv.json = csv.text.toJson();
    json[csv.name] = JSON.parse(csv.json);
  });
  evil(json, data.code);
};
</script>

<template>
  <div>
    <h1>CSV Report</h1>
    <CsvTextArea v-for="csv in data.csvTextList"
      :name="csv.name"
      :text="csv.text" />
    <CodeExitor v-model="data.code" />
    <Button label="run" :onclick="onRun"></Button>
  </div>
</template>

<style scoped>

</style>
