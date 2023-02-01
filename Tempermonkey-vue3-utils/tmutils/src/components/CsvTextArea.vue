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
import type { ICsvTextAreaViewModel } from '@/typings/csv-textarea-models';

let data = reactive<ICsvTextAreaViewModel>({
  csvText: `id,name\n1,flash\n2,jack`,
});
</script>

<template>
  <div>
  </div>
</template>

<style scoped>

</style>
